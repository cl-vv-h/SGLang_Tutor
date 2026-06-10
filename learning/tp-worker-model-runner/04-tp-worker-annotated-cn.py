# -*- coding: utf-8 -*-
# =============================================================================
# SGLang 教学注释版源码副本
# =============================================================================
# 说明：
# 1. 本文件由 learning 教学材料生成，来源于 SGLang 原始源码。
# 2. 这里添加了中文教学注释，帮助理解架构、数据流和关键代码块。
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


# 教学注释：这一层只定义 Scheduler 需要的 worker 能力；真正实现放在 TpModelWorker。
# 教学注释：Tensor Parallel worker 的抽象基类，规定 Scheduler 能调用的模型执行接口，并把大量能力委托给 ModelRunner。
class BaseTpWorker(ABC):
    @abstractmethod
    # 教学注释：生成主入口：把 ScheduleBatch 转成 ForwardBatch，调用 ModelRunner 做前向，并在 PP 末级完成采样。
    def forward_batch_generation(self, forward_batch: ForwardBatch):
        pass

    @property
    @abstractmethod
    # 教学注释：抽象属性或属性方法，用来暴露当前 worker 持有的底层 ModelRunner。
    def model_runner(self) -> "ModelRunner":
        pass

    @property
    # 教学注释：读取模型滑动窗口大小，调度层会据此判断 KV cache 与 attention 的可用范围。
    def sliding_window_size(self) -> Optional[int]:
        return self.model_runner.sliding_window_size

    @property
    # 教学注释：判断模型是否混合使用 full attention 与 sliding-window attention。
    def is_hybrid_swa(self) -> bool:
        return self.model_runner.is_hybrid_swa

    # 教学注释：读取每层 token/KV cache 相关信息，用于容量估算或监控。
    def get_tokens_per_layer_info(self):
        return (
            self.model_runner.full_max_total_num_tokens,
            self.model_runner.swa_max_total_num_tokens,
        )

    # 教学注释：返回 tokenizer/model 相关的 padding 函数，供 batch 构造阶段使用。
    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    # 教学注释：返回 request/token/KV cache 等内存池对象，供调度层观察和复用。
    def get_memory_pool(self) -> Tuple[ReqToTokenPool, BaseTokenToKVPoolAllocator]:
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    # 教学注释：`update_weights_from_disk` 属于运行时热更新/LoRA/权重管理接口，worker 层通常转发，ModelRunner 层负责具体执行。
    def update_weights_from_disk(self, recv_req: UpdateWeightFromDiskReqInput):
        success, message = self.model_runner.update_weights_from_disk(
            recv_req.model_path,
            recv_req.load_format,
            recapture_cuda_graph=recv_req.recapture_cuda_graph,
        )
        return success, message

    # 教学注释：初始化权重更新通信组，让多个 rank 能同步接收更新。
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

    # 教学注释：销毁权重更新通信组，释放相关分布式资源。
    def destroy_weights_update_group(self, recv_req: DestroyWeightsUpdateGroupReqInput):
        success, message = self.model_runner.destroy_weights_update_group(
            recv_req.group_name,
        )
        return success, message

    # 教学注释：`init_weights_send_group_for_remote_instance` 属于初始化阶段，重点看它创建哪些运行时状态以及这些状态会被哪个 forward 分支使用。
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

    # 教学注释：`send_weights_to_remote_instance` 是当前文件中的辅助代码块，建议结合调用点阅读它如何服务 TpModelWorker 与 ModelRunner 的主流程。
    def send_weights_to_remote_instance(
        self, recv_req: SendWeightsToRemoteInstanceReqInput
    ):
        success, message = self.model_runner.send_weights_to_remote_instance(
            recv_req.master_address,
            recv_req.ports,
            recv_req.group_name,
        )
        return success, message

    # 教学注释：通过分布式通信更新权重，适配多 rank 场景。
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

    # 教学注释：从 tensor payload 更新权重，常用于本地或远端权重热更新。
    def update_weights_from_tensor(self, recv_req: UpdateWeightsFromTensorReqInput):

        monkey_patch_torch_reductions()
        success, message = self.model_runner.update_weights_from_tensor(
            named_tensors=MultiprocessingSerializer.deserialize(
                recv_req.serialized_named_tensors[self.tp_rank]
            ),
            load_format=recv_req.load_format,
        )
        return success, message

    # 教学注释：`update_weights_from_ipc` 属于运行时热更新/LoRA/权重管理接口，worker 层通常转发，ModelRunner 层负责具体执行。
    def update_weights_from_ipc(self, recv_req: UpdateWeightsFromIPCReqInput):
        """Update weights from IPC for checkpoint-engine integration."""
        success, message = self.model_runner.update_weights_from_ipc(recv_req)
        return success, message

    # 教学注释：按参数名读取权重，常用于校验、debug 或远端更新流程。
    def get_weights_by_name(self, recv_req: GetWeightsByNameReqInput):
        parameter = self.model_runner.get_weights_by_name(
            recv_req.name, recv_req.truncate_size
        )
        return parameter

    # 教学注释：`load_lora_adapter` 处理模型/权重的加载或保存，是执行层生命周期管理的一部分。
    def load_lora_adapter(self, recv_req: LoadLoRAAdapterReqInput):
        result = self.model_runner.load_lora_adapter(recv_req.to_ref())
        return result

    # 教学注释：`unload_lora_adapter` 是当前文件中的辅助代码块，建议结合调用点阅读它如何服务 TpModelWorker 与 ModelRunner 的主流程。
    def unload_lora_adapter(self, recv_req: UnloadLoRAAdapterReqInput):
        result = self.model_runner.unload_lora_adapter(recv_req.to_ref())
        return result

    # 教学注释：`load_lora_adapter_from_tensors` 处理模型/权重的加载或保存，是执行层生命周期管理的一部分。
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

    # 教学注释：embedding 模型入口：执行 forward 后直接返回 embedding，而不是采样 token。
    def forward_batch_embedding(self, batch: ScheduleBatch):
        # 教学注释：ScheduleBatch 在进入模型前会被转换成 ForwardBatch，补齐位置、KV cache、sampling 等张量视图。
        forward_batch = ForwardBatch.init_new(batch, self.model_runner)
        output = self.model_runner.forward(forward_batch).logits_output
        return output  # Returns EmbeddingPoolerOutput


# 教学注释：TpModelWorker 是每个 TP rank 上的执行入口，生命周期与 server worker 进程绑定。
# 教学注释：真实的 TP 模型 worker，负责按 rank 初始化模型运行器、处理 PP/TP/Spec/dLLM 分支，并承接生成请求。
class TpModelWorker(BaseTpWorker):
    """A tensor parallel model worker."""

    # 教学注释：初始化当前对象的基础状态；在 ModelRunner 中还会触发分布式、模型加载、内存池和 graph 初始化。
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
        # 教学注释：server_args 是运行时总开关，后续分布式、attention backend、graph、LoRA、spec 都依赖它。
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
                self.processor = get_processor(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    tokenizer_backend=server_args.tokenizer_backend,
                )
                self.tokenizer = get_tokenizer_from_processor(self.processor)
            else:
                self.tokenizer = get_tokenizer(
                    server_args.tokenizer_path,
                    tokenizer_mode=server_args.tokenizer_mode,
                    trust_remote_code=server_args.trust_remote_code,
                    revision=server_args.revision,
                    tokenizer_backend=server_args.tokenizer_backend,
                )
        self.device = self.model_runner.device

        # Init nccl groups
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

    # 教学注释：根据当前 worker 是否为 draft worker 选择 target/draft 模型路径并创建 ModelConfig。
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

    # 教学注释：构造 ModelRunner，把分布式 rank、内存池和模型配置交给底层执行器。
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

    # 教学注释：为 multi-layer EAGLE speculative decoding 创建多个 draft ModelRunner。
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

    # 教学注释：按配置初始化 diffusion/denoising LLM 相关算法组件。
    def _init_dllm_algorithm(self):
        from sglang.srt.dllm.algorithm.base import DllmAlgorithm

        if self.server_args.dllm_algorithm is not None:
            self.dllm_algorithm = DllmAlgorithm.from_server_args(self.server_args)
        else:
            self.dllm_algorithm = None

    @property
    # 教学注释：抽象属性或属性方法，用来暴露当前 worker 持有的底层 ModelRunner。
    def model_runner(self) -> "ModelRunner":
        return self._model_runner

    # 教学注释：注册 HiCache 层间传输计数器，便于监控 cache 迁移。
    def register_hicache_layer_transfer_counter(self, counter: LayerDoneCounter):
        self.hicache_layer_transfer_counter = counter

    # 教学注释：为当前 batch 设置 HiCache consumer，使后续 forward 能写入/读取对应 cache 通道。
    def set_hicache_consumer(self, consumer_index: int):
        if self.hicache_layer_transfer_counter is not None:
            self.hicache_layer_transfer_counter.set_consumer(consumer_index)

    # 教学注释：`register_hisparse_coordinator` 是当前文件中的辅助代码块，建议结合调用点阅读它如何服务 TpModelWorker 与 ModelRunner 的主流程。
    def register_hisparse_coordinator(self, coordinator):
        self.model_runner.hisparse_coordinator = coordinator

    # 教学注释：向 Scheduler 暴露 worker 的运行时能力和容量信息。
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

    # 教学注释：判断当前 worker 是否走 diffusion/denoising LLM 路径。
    def is_dllm(self):
        return self.dllm_algorithm is not None

    # 教学注释：dLLM 算法专用生成路径：由 dLLM runner 接管 logits 计算和采样。
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

    # 教学注释：生成主入口：把 ScheduleBatch 转成 ForwardBatch，调用 ModelRunner 做前向，并在 PP 末级完成采样。
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

        # 教学注释：只有 pipeline parallel 的最后一级拥有最终 logits，因此只有最后一级会采样并返回 token。
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

            if is_verify:
                # Skip sampling; spec_v2 worker fires its own publish post-verify.
                return batch_result

            if (
                self.enable_overlap
                and not self.enable_spec
                and forward_batch.sampling_info.grammars is not None
            ):

                # 教学注释：overlap 模式下可以把采样包装成延迟函数，让调度侧与后续工作重叠。
                # 教学注释：`sample_batch_func` 是当前文件中的辅助代码块，建议结合调用点阅读它如何服务 TpModelWorker 与 ModelRunner 的主流程。
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
                # 教学注释：非 PP 末级不采样，只把 hidden states 代理张量交给下一级 pipeline rank。
                pp_hidden_states_proxy_tensors=pp_proxy_tensors,
                can_run_cuda_graph=can_run_cuda_graph,
                expert_distribution_metrics=out.expert_distribution_metrics,
            )

    # 教学注释：chunked/split prefill 的入口：把一次较大的 prefill 拆成多个 forward 片段执行。
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
