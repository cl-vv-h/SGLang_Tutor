# -*- coding: utf-8 -*-
# =============================================================================
# SGLang 中文精读注释版源码副本
# =============================================================================
# 说明：
# 1. 本文件由 learning 教学材料生成，来源于 SGLang 原始源码。
# 2. 注释重点解释每个类、函数和关键代码块在运行时承担的职责。
# 3. 这不是运行时代码，不要从业务代码中 import 本文件。
# 4. 原始源码没有被修改；如需对照，请查看 python/sglang/srt/... 下的对应文件。
# =============================================================================


# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A tensor parallel worker."""

# 下面开始保留原始 imports。
# 这些 import 展示了该文件依赖的边界：tp_worker 主要依赖 manager、distributed 和 ModelRunner；model_runner 则依赖分布式、attention、KV cache、graph、loader、sampling 等几乎整个执行层。
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from sglang.srt.distributed import get_pp_group, get_world_group
from sglang.srt.managers.io_struct import (
    DestroyWeightsUpdateGroupReqInput,
    GetWeightsByNameReqInput,
    InitWeightsSendGroupForRemoteInstanceReqInput,
    InitWeightsUpdateGroupReqInput,
    LoadLoRAAdapterFromTensorsReqInput,
    LoadLoRAAdapterReqInput,
    SendWeightsToRemoteInstanceReqInput,
    UnloadLoRAAdapterReqInput,
    UpdateWeightFromDiskReqInput,
    UpdateWeightsFromDistributedReqInput,
    UpdateWeightsFromIPCReqInput,
    UpdateWeightsFromTensorReqInput,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import MultiprocessingSerializer, broadcast_pyobj, set_random_seed
from sglang.srt.utils.hf_transformers_utils import (
    get_processor,
    get_tokenizer,
    get_tokenizer_from_processor,
)
from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

if TYPE_CHECKING:
    from sglang.srt.managers.cache_controller import LayerDoneCounter
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig

logger = logging.getLogger(__name__)


# BaseTpWorker 是 Scheduler 能看到的 TP worker 抽象接口。
# 它不关心具体模型如何加载、forward 如何执行，只规定 worker 必须提供哪些能力。
# 这里很多方法只是把调用转交给 ModelRunner，目的是让调度层不直接依赖模型执行细节。
class BaseTpWorker(ABC):
    @abstractmethod
    # 这是 TpModelWorker 的生成主入口。
    # 输入是 Scheduler 组织好的 ScheduleBatch；函数先构造 ForwardBatch，再调用 ModelRunner.forward。
    # 如果当前 PP rank 是最后一级，它会拿到 logits 并执行 sampling；否则只把 hidden states 传给下一段 pipeline。
    # 这里也是 speculative verify、prefill-only、overlap sampling、dLLM 等分支汇合的位置。
    def forward_batch_generation(self, forward_batch: ForwardBatch):
        pass

    @property
    @abstractmethod
    # 暴露当前 worker 持有的 ModelRunner。
    # BaseTpWorker 通过抽象属性约束子类；Scheduler 和通用 worker 方法可以统一访问底层执行器。
    def model_runner(self) -> "ModelRunner":
        pass

    @property
    # 返回模型的 sliding window 大小。
    # 调度层和 KV cache 逻辑会根据这个值判断哪些历史 token 仍需保留在 attention 窗口内。
    def sliding_window_size(self) -> Optional[int]:
        return self.model_runner.sliding_window_size

    @property
    # 返回模型是否混合使用 full attention 与 sliding-window attention。
    # 如果为 true，调度和 KV cache 容量估算需要同时考虑普通层与 SWA 层的 token 上限。
    def is_hybrid_swa(self) -> bool:
        return self.model_runner.is_hybrid_swa

    # 返回 full attention 层和 SWA 层分别可容纳的 token 数。
    # hybrid SWA 模型中，不同层的 KV cache 容量可能不同，调度层需要这两个值做安全容量判断。
    def get_tokens_per_layer_info(self):
        return (
            self.model_runner.full_max_total_num_tokens,
            self.model_runner.swa_max_total_num_tokens,
        )

    # 返回模型自定义的 pad_input_ids 函数。
    # 有些模型在 padding input ids 时需要特殊 token 或位置处理；没有自定义函数时返回 None。
    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    # 暴露请求到 token 的映射池，以及 token 到 KV cache 的分配器。
    # Scheduler 通过这些池管理 continuous batching 中每个请求占用的 KV cache 位置。
    def get_memory_pool(self) -> Tuple[ReqToTokenPool, BaseTokenToKVPoolAllocator]:
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    # 从磁盘路径加载新的权重并更新当前模型。
    # worker 层负责接收请求对象，ModelRunner 层负责真正调用 loader、同步 rank 并刷新模型状态。
    # 这类接口常用于不停服替换权重或把模型切换到新的 checkpoint。
    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.model_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        return success, message

    # 初始化权重更新使用的分布式通信组。
    # 多 rank 模型在热更新时必须让每个 rank 收到自己负责的权重分片，因此需要单独的更新组来协调通信。
    def init_weights_update_group(self, recv_req: InitWeightsUpdateGroupReqInput):
        success, message = self.model_runner.init_weights_update_group(
            recv_req.master_address,
            recv_req.master_port,
            recv_req.rank_offset,
            recv_req.world_size,
            recv_req.group_name,
            recv_req.backend,
        )
        return success, message

    # 销毁权重更新通信组。
    # 当一次热更新流程结束或不再需要该 group 时释放资源，避免长期占用分布式通信句柄。
    def destroy_weights_update_group(self, recv_req: DestroyWeightsUpdateGroupReqInput):
        success, message = self.model_runner.destroy_weights_update_group(
            recv_req.group_name,
        )
        return success, message

    # 初始化向远端实例发送权重所需的通信组。
    # 这个路径用于把当前实例的权重同步给另一个远端服务实例，常见于模型迁移、实例扩容或远端热更新。
    def init_weights_send_group_for_remote_instance(
        self, recv_req: InitWeightsSendGroupForRemoteInstanceReqInput
    ):
        success, message = (
            self.model_runner.init_weights_send_group_for_remote_instance(
                recv_req.master_address,
                recv_req.ports,
                recv_req.group_rank,
                recv_req.world_size,
                recv_req.group_name,
                recv_req.backend,
            )
        )
        return success, message

    # 把当前模型权重发送到远端实例。
    # 函数会依赖前面建立的 send group，把每个 rank 持有的参数分片传到对应接收端。
    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        success, message = self.model_runner.send_weights_to_remote_instance(
            recv_req.master_address,
            recv_req.ports,
            recv_req.group_name,
        )
        return success, message

    # 通过分布式通信接收并更新权重。
    # 这条路径适合权重已经分散在多个 rank 或远端发送端的场景，避免把完整权重先聚合到单个进程。
    def update_weights_from_distributed(
        self, recv_req: UpdateWeightsFromDistributedReqInput
    ):
        success, message = self.model_runner.update_weights_from_distributed(
            recv_req.names,
            recv_req.dtypes,
            recv_req.shapes,
            recv_req.group_name,
            recv_req.load_format,
        )
        return success, message

    # 从内存中的 tensor 字典更新权重。
    # 这条路径绕过磁盘文件，适合控制面已经把参数 tensor 直接传给 worker 的情况。
    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):

        monkey_patch_torch_reductions()
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=MultiprocessingSerializer.deserialize(
                recv_req.serialized_named_tensors[self.tp_rank]
            ),
            load_format=recv_req.load_format,
        )
        return success, message

    # 通过 IPC 共享内存或句柄接收权重 tensor。
    # 这种方式避免大 tensor 在进程间重复拷贝，适合本机多进程权重热更新。
    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update weights from IPC for checkpoint-engine integration."""
        success, message = self.model_runner.update_weights_from_ipc(recv_req)
        return success, message

    # 按参数名读取当前模型权重。
    # 常用于热更新前后的校验、debug，或把本实例的某些权重片段发送给远端实例。
    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.model_runner.get_weights_by_name(
            recv_req.name, recv_req.truncate_size
        )
        return parameter

    # 动态加载一个 LoRA adapter。
    # ModelRunner 会把 LoRA 权重放入 LoRA manager 管理；后续请求可以通过 lora id 选择使用哪个 adapter。
    def load_lora_adapter(self, recv_req: LoadLoRAAdapterReqInput):
        result = self.model_runner.load_lora_adapter(recv_req.to_ref())
        return result

    # 卸载一个 LoRA adapter。
    # 这会从 LoRA manager 中移除对应 adapter 的权重和索引，后续请求不能再引用它。
    def unload_lora_adapter(self, recv_req: UnloadLoRAAdapterReqInput):
        result = self.model_runner.unload_lora_adapter(recv_req.to_ref())
        return result

    # 直接从 tensor payload 加载 LoRA adapter。
    # 如果 payload 是 flattened_bucket，会先根据 metadata 还原各个 LoRA tensor，再交给 LoRA manager 注册。
    def load_lora_adapter_from_tensors(
        self, recv_req: LoadLoRAAdapterFromTensorsReqInput
    ):
        # The LoRA code handles TP sharding internally using slice_lora_a_weights
        # and slice_lora_b_weights methods (see lora/layers.py:46-49, mem_pool.py:437-440).
        if recv_req.load_format == "flattened_bucket":
            flattened_data = MultiprocessingSerializer.deserialize(
                recv_req.serialized_tensors
            )
            bucket = FlattenedTensorBucket(
                flattened_tensor=flattened_data["flattened_tensor"],
                metadata=flattened_data["metadata"],
            )
            tensors = dict(bucket.reconstruct_tensors())
        else:
            tensors = MultiprocessingSerializer.deserialize(recv_req.serialized_tensors)
        result = self.model_runner.load_lora_adapter_from_tensors(
            recv_req.to_ref(),
            tensors,
            recv_req.config_dict,
            recv_req.added_tokens_config,
        )
        return result

    # embedding 模型不需要采样 next token。
    # 这里仍然复用 ForwardBatch 和 ModelRunner.forward，但直接返回 embedding/pooler 输出。
    def forward_batch_embedding(self, batch: ScheduleBatch):
        # 把调度层 ScheduleBatch 转换为模型层 ForwardBatch。
        # 这个转换会准备 input_ids、positions、seq_lens、KV cache location、sampling_info、spec_info 等模型执行需要的张量视图。
        forward_batch = ForwardBatch.init_new(batch, self.model_runner)
        output = self.model_runner.forward(forward_batch).logits_output
        return output  # Returns EmbeddingPoolerOutput


# TpModelWorker 是每个 tensor-parallel rank 上实际运行的 worker。
# 它负责把 server_args、TP/PP rank、模型配置和 tokenizer 等运行时信息组装起来。
# 请求到来时，它把 ScheduleBatch 转成 ForwardBatch，再调用 ModelRunner 执行真正的模型前向。
# 如果启用了 PP、speculative decoding、dLLM 或 overlap，这一层会决定走哪条分支。
class TpModelWorker(BaseTpWorker):
    """A tensor parallel model worker."""

    # 初始化当前对象持有的运行时状态。
    # 在 TpModelWorker 中，这里会创建 ModelConfig、ModelRunner、tokenizer 和并行通信组引用。
    # 在 ModelRunner 中，这里会保存设备/rank/spec/parallel 配置，并继续触发分布式初始化与模型执行环境初始化。
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        moe_ep_rank: int,
        pp_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        dp_rank: Optional[int],
        nccl_port: int,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        is_multi_layer_eagle: bool = False,
    ):
        # Parse args
        # 保存 server_args。
        # 这是运行时配置的总入口，后续是否启用 speculative decoding、LoRA、CUDA graph、attention backend、MoE、DP/PP/TP 等都从这里读取。
        self.server_args = server_args
        self.tp_size = server_args.tp_size
        self.ep_size = server_args.ep_size
        self.pp_size = server_args.pp_size
        self.tp_rank = tp_rank
        self.moe_ep_rank = moe_ep_rank
        self.pp_rank = pp_rank
        self.dp_rank = dp_rank
        self.gpu_id = gpu_id
        self.nccl_port = nccl_port
        self.is_draft_worker = is_draft_worker
        self.is_multi_layer_eagle = is_multi_layer_eagle
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        # Draft worker: target's resolved MemoryPoolConfig (forwarded to ModelRunner).
        self.memory_pool_config = memory_pool_config

        # MTP model runners
        self.model_runner_list: List[ModelRunner] = []

        self._init_model_config()
        self._init_model_runner()

        if is_multi_layer_eagle:
            self._init_multi_layer_eagle_model_runners()

        self._init_dllm_algorithm()

        if server_args.skip_tokenizer_init:
            self.tokenizer = self.processor = None
        else:
            if self.model_config.is_multimodal:
                # 初始化 processor。
                # 多模态模型可能需要 processor 处理图像、视频或其他非文本输入。
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    tokenizer_backend=server_args.tokenizer_backend,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                # 初始化 tokenizer。
                # worker 需要 tokenizer 来处理 padding、特殊 token、stop token 等与请求解析相关的逻辑。
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    tokenizer_backend=server_args.tokenizer_backend,
                )
        self.device = self.model_runner.device

        # Init nccl groups
        # 读取 pipeline parallel 通信组。
        # 后续 forward_batch_generation 会用 pp_group.is_last_rank 判断当前 rank 是否负责采样。
        self.pp_group = get_pp_group()
        self.world_group = get_world_group()

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = self.model_runner.max_running_requests
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_queued_requests = server_args.max_queued_requests
        assert (
            self.max_queued_requests is None or self.max_queued_requests >= 1
        ), "If configured, max_queued_requests must be at least 1 for any work to be scheduled."
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.model_runner.max_token_pool_size - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert (
            self.max_req_len > 0 and self.max_req_input_len > 0
        ), "Memory pool size is too small"

        # Sync random seed across TP workers
        self.random_seed = broadcast_pyobj(
            [server_args.random_seed],
            self.tp_size * self.pp_rank + tp_rank,
            self.world_group.cpu_group,
            src=self.world_group.ranks[0],
        )[0]
        set_random_seed(self.random_seed)

        self.enable_overlap = not server_args.disable_overlap_schedule
        self.enable_spec = server_args.speculative_algorithm is not None
        self.hicache_layer_transfer_counter = None

    # 根据 worker 身份选择模型路径并创建 ModelConfig。
    # 普通 target worker 使用主模型路径；draft worker 使用 speculative_draft_model_path。
    # 这样 speculative decoding 可以让 target 模型和 draft 模型拥有不同配置。
    def _init_model_config(self):
        from sglang.srt.configs.model_config import ModelConfig

        self.model_config = ModelConfig.from_server_args(
            self.server_args,
            model_path=(
                self.server_args.model_path
                if not self.is_draft_worker
                else self.server_args.speculative_draft_model_path
            ),
            model_revision=(
                self.server_args.revision
                if not self.is_draft_worker
                else self.server_args.speculative_draft_model_revision
            ),
            is_draft_model=self.is_draft_worker,
        )

    # 创建底层 ModelRunner。
    # 这里把 TP rank、PP rank、DP rank、GPU id、通信端口和内存池等信息一起传入。
    # 从这一刻开始，模型加载、KV cache、attention backend 等重资源都交给 ModelRunner 管理。
    def _init_model_runner(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        self._model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=self.server_args.mem_fraction_static,
            gpu_id=self.gpu_id,
            tp_rank=self.tp_rank,
            tp_size=self.tp_size,
            moe_ep_rank=self.moe_ep_rank,
            moe_ep_size=self.ep_size,
            pp_rank=self.pp_rank,
            pp_size=self.pp_size,
            nccl_port=self.nccl_port,
            dp_rank=self.dp_rank,
            server_args=self.server_args,
            is_draft_worker=self.is_draft_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            memory_pool_config=self.memory_pool_config,
            draft_model_idx=0 if self.is_multi_layer_eagle else None,
        )

    # multi-layer EAGLE 会在不同 draft step 使用多个 draft runner。
    # 这个函数按 speculative_num_steps 创建一组 ModelRunner，让每一步草稿验证可以拥有独立执行状态。
    def _init_multi_layer_eagle_model_runners(self):
        from sglang.srt.model_executor.model_runner import ModelRunner

        self.model_runner_list.append(self.model_runner)
        for i in range(1, self.server_args.speculative_num_steps):
            self.model_runner_list.append(
                ModelRunner(
                    model_config=self.model_config,
                    mem_fraction_static=self.server_args.mem_fraction_static,
                    gpu_id=self.gpu_id,
                    tp_rank=self.tp_rank,
                    tp_size=self.tp_size,
                    moe_ep_rank=self.moe_ep_rank,
                    moe_ep_size=self.ep_size,
                    pp_rank=self.pp_rank,
                    pp_size=self.pp_size,
                    nccl_port=self.nccl_port,
                    dp_rank=self.dp_rank,
                    server_args=self.server_args,
                    is_draft_worker=self.is_draft_worker,
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    memory_pool_config=self.memory_pool_config,
                    draft_model_idx=i,
                )
            )

    # dLLM 是 diffusion/denoising 风格的生成路径。
    # 如果配置启用 dLLM，这里会初始化对应算法对象；后续 forward_batch_generation 会分流到 dLLM 专用实现。
    def _init_dllm_algorithm(self):
        from sglang.srt.dllm.algorithm.base import DllmAlgorithm

        if self.server_args.dllm_algorithm is not None:
            self.dllm_algorithm = DllmAlgorithm.from_server_args(self.server_args)
        else:
            self.dllm_algorithm = None

    @property
    # 暴露当前 worker 持有的 ModelRunner。
    # BaseTpWorker 通过抽象属性约束子类；Scheduler 和通用 worker 方法可以统一访问底层执行器。
    def model_runner(self) -> "ModelRunner":
        return self._model_runner

    # 注册 HiCache 层级传输计数器。
    # 计数器用于观察每层 cache 传输完成情况，帮助调度或诊断跨层缓存迁移。
    def register_hicache_layer_transfer_counter(self, counter: LayerDoneCounter):
        self.hicache_layer_transfer_counter = counter

    # 为当前 batch 设置 HiCache consumer 索引。
    # 后续 forward 中的 cache 读取/写入会带上这个 consumer 标识，以区分不同请求或不同缓存通道。
    def set_hicache_consumer(self, consumer_index: int):
        if self.hicache_layer_transfer_counter is not None:
            self.hicache_layer_transfer_counter.set_consumer(consumer_index)

    # 注册 HiSparse 协调器。
    # HiSparse 相关 attention 或 cache 路径需要一个协调对象来管理稀疏结构和跨层状态。
    def register_hisparse_coordinator(self, coordinator):
        self.model_runner.hisparse_coordinator = coordinator

    # 把当前 worker 的容量和能力返回给调度层。
    # Scheduler 需要知道 max_total_num_tokens、max_running_requests、模型配置和并行状态，才能做 batch 规划。
    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_queued_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            self.model_runner.forward_stream,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    # 判断当前 worker 是否启用了 dLLM 算法对象。
    # 如果返回 true，生成入口会走 _forward_batch_generation_dllm，而不是普通自回归 forward/sampling 路径。
    def is_dllm(self):
        return self.dllm_algorithm is not None

    # dLLM 专用生成路径，不走普通 autoregressive sampling 分支。
    # 它调用 dLLM runner 产生 logits 和采样结果，再包装成 GenerationBatchResult 返回给调度层。
    def _forward_batch_generation_dllm(
        self, forward_batch: ForwardBatch
    ) -> GenerationBatchResult:
        logits_output, next_token_ids, can_run_cuda_graph = self.dllm_algorithm.run(
            self.model_runner, forward_batch
        )
        return GenerationBatchResult(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    # 这是 TpModelWorker 的生成主入口。
    # 输入是 Scheduler 组织好的 ScheduleBatch；函数先构造 ForwardBatch，再调用 ModelRunner.forward。
    # 如果当前 PP rank 是最后一级，它会拿到 logits 并执行 sampling；否则只把 hidden states 传给下一段 pipeline。
    # 这里也是 speculative verify、prefill-only、overlap sampling、dLLM 等分支汇合的位置。
    def forward_batch_generation(
        self,
        batch: Optional[ScheduleBatch],
        forward_batch: Optional[ForwardBatch] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        is_verify: bool = False,
        skip_attn_backend_init=False,
    ) -> GenerationBatchResult:
        # FIXME(lsyin): maybe remove skip_attn_backend_init in forward_batch_generation,
        #               which requires preparing replay to always be in this function
        # Get forward batch from schedule batch

        if batch is not None:
            # update the consumer index of hicache to the running batch
            self.set_hicache_consumer(batch.hicache_consumer_index)

            forward_batch = ForwardBatch.init_new(batch, self.model_runner)
        else:
            # FIXME(lsyin): unify the interface of forward_batch
            assert forward_batch is not None

        if self.is_dllm():
            return self._forward_batch_generation_dllm(forward_batch)

        # 判断当前 pipeline stage 是否是最后一级。
        # 只有最后一级能看到最终 logits，因此只有它会执行 sampling 并返回 next_token_ids。
        # 非最后一级只负责计算 hidden states 并把 proxy tensor 传给下一级。
        if self.pp_group.is_last_rank:
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            batch_result = GenerationBatchResult(
                logits_output=logits_output,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
                routed_experts_output=out.routed_experts_output,
                indexer_topk_output=out.indexer_topk_output,
            )

            # speculative decoding 的 verify 阶段不一定立即采样。
            # 这里直接返回 logits/hidden states，让上层验证 draft token 是否被 target 模型接受。
            if is_verify:
                # Skip sampling; spec_v2 worker fires its own publish post-verify.
                return batch_result

            if (
                self.enable_overlap
                and not self.enable_spec
                and forward_batch.sampling_info.grammars is not None
            ):

                # overlap 模式下的延迟采样闭包。
                # 外层先返回一个可调用对象，调度侧可以在合适时机执行它，从而把采样和后续调度工作重叠。
                def sample_batch_func():
                    batch_result.next_token_ids = self.model_runner.sample(
                        logits_output, forward_batch
                    )
                    return batch_result

                batch_result.delay_sample_func = sample_batch_func
                return batch_result

            if not forward_batch.is_prefill_only:
                # For normal requests, sample the next token ids.
                batch_result.next_token_ids = self.model_runner.sample(
                    logits_output, forward_batch
                )
            else:
                # For prefill-only requests, create dummy token IDs on CPU
                # The size should match the batch size (number of sequences), not total tokens
                batch_result.next_token_ids = torch.zeros(
                    len(forward_batch.seq_lens),
                    dtype=torch.long,
                    device=forward_batch.input_ids.device,
                )
                if (
                    forward_batch.return_logprob
                    and logits_output.next_token_logits is not None
                ):
                    # NOTE: Compute logprobs without full sampling
                    self.model_runner.compute_logprobs_only(
                        logits_output, forward_batch
                    )

            return batch_result
        else:
            out = self.model_runner.forward(
                forward_batch,
                pp_proxy_tensors=pp_proxy_tensors,
                skip_attn_backend_init=skip_attn_backend_init,
            )
            pp_proxy_tensors, can_run_cuda_graph = out.logits_output, out.can_run_graph
            return GenerationBatchResult(
                pp_hidden_states_proxy_tensors=pp_proxy_tensors,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
            )

    # split prefill 会把长 prompt 的 prefill 分成多个小片段执行。
    # 第一个 split 构造 ForwardBatch，后续 split 复用同一个对象并推进 split_index。
    # 只有某个 split 产生最终 logits 时才执行 sampling，否则继续等待下一片段。
    def forward_batch_split_prefill(self, batch: ScheduleBatch):
        if batch.split_index == 0:
            forward_batch = ForwardBatch.init_new(batch, self.model_runner)
            batch.split_forward_batch = forward_batch

        out = self.model_runner.forward(
            batch.split_forward_batch, split_forward_count=batch.split_forward_count
        )
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        if logits_output:
            next_token_ids = self.model_runner.sample(
                logits_output, batch.split_forward_batch
            )
        else:
            next_token_ids = None
        batch_result = GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=can_run_cuda_graph,
            expert_distribution_metrics=out.expert_distribution_metrics,
        )
        batch_result.next_token_ids = next_token_ids
        return batch_result
