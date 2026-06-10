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
"""ModelRunner runs the forward passes of the models."""

# 下面开始保留原始 imports。
# 这些 import 展示了该文件依赖的边界：tp_worker 主要依赖 manager、distributed 和 ModelRunner；model_runner 则依赖分布式、attention、KV cache、graph、loader、sampling 等几乎整个执行层。
from __future__ import annotations

import contextlib
import datetime
import gc
import hashlib
import inspect
import logging
import os
import socket
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch import nn

from sglang.jit_kernel.ngram_embedding import update_token_table
from sglang.srt.configs import (
    BailingHybridConfig,
    FalconH1Config,
    GraniteMoeHybridConfig,
    InternS2PreviewConfig,
    JetNemotronConfig,
    JetVLMConfig,
    KimiLinearConfig,
    Lfm2Config,
    Lfm2MoeConfig,
    Lfm2VlConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3NextConfig,
)
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_config
from sglang.srt.configs.load_config import LoadConfig, LoadFormat
from sglang.srt.configs.model_config import (
    AttentionArch,
    ModelConfig,
    ModelImpl,
    get_num_indexer_layers,
)
from sglang.srt.configs.update_config import adjust_config_with_unaligned_cpu_tp
from sglang.srt.constants import GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.debug_utils.dumper import dumper
from sglang.srt.debug_utils.tensor_dump_forward_hook import (
    register_forward_hook_for_model,
)
from sglang.srt.distributed import (
    get_default_distributed_backend,
    get_pp_group,
    get_tp_group,
    get_world_group,
    # 初始化 torch distributed 进程组。
    # 这是 NCCL/Gloo/Mooncake 等通信 backend 的基础，所有 TP/PP/DP collective 都依赖它。
    init_distributed_environment,
    # 创建 SGLang/vLLM 风格的模型并行组。
    # 这里同时配置 TP、PP、attention DP/CP、MoE EP/DP 等维度，使后续模型层可以直接查询所属 group。
    initialize_model_parallel,
    set_custom_all_reduce,
    set_mscclpp_all_reduce,
    set_torch_symm_mem_all_reduce,
)
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import monkey_patch_vllm_parallel_state
from sglang.srt.elastic_ep.elastic_ep import (
    ElasticEPStateManager,
    join_process_groups,
    try_recover_ranks,
)
from sglang.srt.elastic_ep.expert_backup_client import ExpertBackupClient
from sglang.srt.environ import envs
from sglang.srt.eplb.eplb_manager import EPLBManager
from sglang.srt.eplb.expert_distribution import (
    ExpertDistributionMetrics,
    ExpertDistributionRecorder,
    get_global_expert_distribution_recorder,
    set_global_expert_distribution_recorder,
)
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    broadcast_global_expert_location_metadata,
    compute_initial_expert_location_metadata,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from sglang.srt.eplb.expert_location_updater import ExpertLocationUpdater
from sglang.srt.hardware_backend.npu.graph_runner.npu_graph_runner import NPUGraphRunner
from sglang.srt.kv_canary.api import install_canary
from sglang.srt.kv_canary.runner.canary_manager import context_tuple
from sglang.srt.kv_canary.token_oracle.install import install_token_oracle_from_env
from sglang.srt.layers import deep_gemm_wrapper
from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    attn_backend_wrapper,
)
from sglang.srt.layers.attention.dsa.utils import is_dsa_enable_prefill_cp
from sglang.srt.layers.attention.tbo_backend import TboAttnBackend
from sglang.srt.layers.dp_attention import (
    DpPaddingMode,
    get_attention_tp_group,
    get_attention_tp_size,
    initialize_dp_attention,
    set_dp_buffer_len,
    set_is_extend_in_batch,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.moe.hash_topk import HashTopK
from sglang.srt.layers.moe.topk import TopK
from sglang.srt.layers.pooler import EmbeddingPoolerOutput
from sglang.srt.layers.quantization.fp8_kernel import fp8_dtype
from sglang.srt.layers.sampler import create_sampler
from sglang.srt.layers.torchao_utils import apply_torchao_config_to_model
from sglang.srt.layers.utils.cp_utils import is_mla_prefill_cp_enabled
from sglang.srt.lora.lora_manager import LoRAManager
from sglang.srt.lora.lora_registry import LoRARef
from sglang.srt.managers.schedule_batch import sanity_check_mm_pad_shift_value
from sglang.srt.mem_cache.allocator import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.model_executor.breakable_cuda_graph_runner import (
    BreakableCudaGraphRunner,
)
from sglang.srt.model_executor.cpu_graph_runner import CPUGraphRunner
from sglang.srt.model_executor.cuda_graph_runner import (
    CudaGraphRunner,
    DecodeInputBuffers,
    set_torch_compile_config,
)
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
    PPProxyTensors,
)
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    has_forward_context,
)
from sglang.srt.model_executor.hook_manager import register_forward_hooks
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.srt.model_executor.piecewise_cuda_graph_runner import (
    PiecewiseCudaGraphRunner,
)
from sglang.srt.model_executor.pool_configurator import MemoryPoolConfig
from sglang.srt.model_loader.loader import DefaultModelLoader, get_model_loader
from sglang.srt.model_loader.remote_instance_weight_loader_utils import (
    RemoteInstanceWeightLoaderBackend,
    register_memory_region,
    trigger_init_weights_send_group_for_remote_instance_request,
)
from sglang.srt.model_loader.utils import set_default_torch_dtype
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.platforms import current_platform
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.server_args import (
    ServerArgs,
    get_global_server_args,
    set_global_server_args_for_scheduler,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.state_capturer.base import TopkCaptureOutput
from sglang.srt.state_capturer.indexer_topk import (
    create_indexer_capturer,
    get_global_indexer_capturer,
    set_global_indexer_capturer,
)
from sglang.srt.state_capturer.routed_experts import (
    RoutedExpertsCapturer,
    get_global_experts_capturer,
    set_global_experts_capturer,
)
from sglang.srt.utils import (
    MultiprocessingSerializer,
    broadcast_pyobj,
    cpu_has_amx_support,
    dynamic_import,
    empty_context,
    enable_show_time_cost,
    get_available_gpu_memory,
    get_bool_env_var,
    get_cpu_ids_by_node,
    init_custom_process_group,
    is_hip,
    is_host_cpu_arm64,
    is_npu,
    log_info_on_rank0,
    monkey_patch_p2p_access_check,
    require_attn_tp_gather,
    require_gathered_buffer,
    require_mlp_tp_gather,
    reserve_rope_cache_for_long_sequences,
    set_cuda_arch,
    slow_rank_detector,
)
from sglang.srt.utils.common import ceil_align, require_mlp_sync
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto
from sglang.srt.utils.nvtx_pytorch_hooks import PytHooks
from sglang.srt.utils.offloader import (
    create_offloader_from_server_args,
    get_offloader,
    set_offloader,
)
from sglang.srt.utils.patch_torch import (
    monkey_patch_torch_reductions,
    register_sgl_tp_rank,
)
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils.weight_checker import WeightChecker
from sglang.srt.weight_sync.tensor_bucket import (
    FlattenedTensorBucket,
    FlattenedTensorMetadata,
)

_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu_arm64 = is_host_cpu_arm64()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _is_npu:
    from sglang.srt.hardware_backend.npu.utils import init_npu_backend

    init_npu_backend()
elif current_platform.is_out_of_tree():
    current_platform.init_backend()

MLA_ATTENTION_BACKENDS = [
    "aiter",
    "flashinfer",
    "fa3",
    "fa4",
    "triton",
    "flashmla",
    "cutedsl_mla",
    "cutlass_mla",
    "trtllm_mla",
    "tokenspeed_mla",
    "ascend",
    "dsa",
    "nsa",  # Deprecated alias for "dsa"
    "intel_xpu",
]

CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS = [
    "flashinfer",
    "fa3",
    "fa4",
    "flashmla",
    "cutedsl_mla",
    "cutlass_mla",
    "trtllm_mla",
    "tokenspeed_mla",
]

TORCH_DTYPE_TO_KV_CACHE_STR = {
    torch.float8_e4m3fn: "fp8_e4m3",
    torch.float8_e4m3fnuz: "fp8_e4m3",
    torch.float8_e5m2: "fp8_e5m2",
    torch.bfloat16: "bf16",
}


# MLA 模型需要特殊 attention backend。
# 这个函数根据模型配置动态注册 backend，使后续 init_attention_backend 可以按统一名字创建实现。
def add_mla_attention_backend(backend_name):
    if backend_name not in MLA_ATTENTION_BACKENDS:
        MLA_ATTENTION_BACKENDS.append(backend_name)
        logger.info(f"Added {backend_name} to MLA_ATTENTION_BACKENDS.")


# chunked prefix cache 会改变 attention backend 的缓存组织方式。
# 这里在运行时把对应 backend 加入注册表，让配置解析阶段可以选择它。
def add_chunked_prefix_cache_attention_backend(backend_name):
    if backend_name not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS:
        CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS.append(backend_name)
        logger.info(
            f"Added {backend_name} to CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS."
        )


# Detect stragger ranks in model loading
UNBALANCED_MODEL_LOADING_TIMEOUT_S = 480  # leave more time for post data processing


logger = logging.getLogger(__name__)

_UNSET: Any = object()


# 不同模型 wrapper 的层级结构不一致，有的 layers 在 model.model.layers，有的直接在 model.layers。
# piecewise CUDA graph 需要枚举 transformer layers，因此先通过该函数找到真正的语言模型主体。
def resolve_language_model(model: nn.Module) -> nn.Module:
    model_cls_name = model.__class__.__name__
    if model_cls_name == "Qwen3OmniMoeForConditionalGeneration":
        return model.thinker.model
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "language_model"):
        return model.language_model
    return model.model


# 多进程并行启动时，每个 rank 都可能打印同样日志。
# 这个过滤器只允许 rank0 输出部分日志，避免初始化和加载阶段的日志被重复刷屏。
class RankZeroFilter(logging.Filter):
    """Filter that only allows INFO level logs from rank 0, but allows all other levels from any rank."""

    # 初始化当前对象持有的运行时状态。
    # 在 TpModelWorker 中，这里会创建 ModelConfig、ModelRunner、tokenizer 和并行通信组引用。
    # 在 ModelRunner 中，这里会保存设备/rank/spec/parallel 配置，并继续触发分布式初始化与模型执行环境初始化。
    def __init__(self, is_rank_zero):
        super().__init__()
        self.is_rank_zero = is_rank_zero

    # 日志过滤器的实际判断函数。
    # record 只有在当前进程是 rank0 时才会通过，其他 rank 的重复日志会被过滤掉。
    def filter(self, record):
        if record.levelno == logging.INFO:
            return self.is_rank_zero
        return True


@dataclass
# ModelRunner.forward 的统一返回结构。
# logits_output 可能是真正的 logits，也可能是 PP 中间 rank 传给下一阶段的 PPProxyTensors。
# 其余字段用于告诉上层是否可走 graph、是否发生 cuda graph padding，以及 speculative/metrics/debug 相关输出。
class ModelRunnerOutput:
    logits_output: Union[LogitsProcessorOutput, PPProxyTensors]
    can_run_graph: bool
    expert_distribution_metrics: Optional[ExpertDistributionMetrics] = None
    routed_experts_output: Optional[TopkCaptureOutput] = None
    indexer_topk_output: Optional[TopkCaptureOutput] = None


# ModelRunner 是 SGLang 模型执行层的核心对象。
# 它把模型权重、分布式通信组、KV cache 内存池、attention backend、CUDA graph、采样器等运行资源放在一起管理。
# Scheduler 和 TpModelWorker 不直接接触这些底层资源，而是通过 ModelRunner.forward/sample 等统一入口使用它们。
# 阅读这个类时，可以按“初始化资源 -> 构造 ForwardBatch metadata -> 执行 model.forward -> 采样/返回结果”的顺序理解。
class ModelRunner(ModelRunnerKVCacheMixin):
    """ModelRunner runs the forward passes of the models."""

    # 初始化当前对象持有的运行时状态。
    # 在 TpModelWorker 中，这里会创建 ModelConfig、ModelRunner、tokenizer 和并行通信组引用。
    # 在 ModelRunner 中，这里会保存设备/rank/spec/parallel 配置，并继续触发分布式初始化与模型执行环境初始化。
    def __init__(
        self,
        model_config: ModelConfig,
        mem_fraction_static: float,
        gpu_id: int,
        tp_rank: int,
        tp_size: int,
        moe_ep_rank: int,
        moe_ep_size: int,
        pp_rank: int,
        pp_size: int,
        nccl_port: int,
        server_args: ServerArgs,
        dp_rank: Optional[int] = None,
        attn_cp_rank: Optional[int] = None,
        moe_dp_rank: Optional[int] = None,
        is_draft_worker: bool = False,
        req_to_token_pool: Optional[ReqToTokenPool] = None,
        token_to_kv_pool_allocator: Optional[BaseTokenToKVPoolAllocator] = None,
        memory_pool_config: Optional[MemoryPoolConfig] = None,
        draft_model_idx: Optional[int] = None,
    ):
        # Parse args
        self.mem_fraction_static = mem_fraction_static
        # Set on target by `_resolve_memory_pool_config`; passed in for draft
        # workers so they reuse target's resolved sizes (replaces legacy
        # `server_args._draft_pool_config` mutation hack).
        self.memory_pool_config = memory_pool_config
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.moe_ep_rank = moe_ep_rank
        self.moe_ep_size = moe_ep_size
        self.dp_rank = dp_rank
        self.dp_size = server_args.dp_size if server_args.enable_dp_attention else 1
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.attn_cp_rank = attn_cp_rank
        self.attn_cp_size = server_args.attn_cp_size
        self.moe_dp_rank = moe_dp_rank
        self.moe_dp_size = server_args.moe_dp_size
        self.model_config = model_config
        self.dist_port = nccl_port
        # 保存 server_args。
        # 这是运行时配置的总入口，后续是否启用 speculative decoding、LoRA、CUDA graph、attention backend、MoE、DP/PP/TP 等都从这里读取。
        self.server_args = server_args
        self.is_draft_worker = is_draft_worker
        self.is_generation = model_config.is_generation
        self.device_timer = None
        self.is_multimodal = model_config.is_multimodal
        self.is_multimodal_chunked_prefill_supported = (
            model_config.is_multimodal_chunked_prefill_supported
        )
        self.spec_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.page_size = server_args.page_size
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.is_hybrid_swa = model_config.is_hybrid_swa
        self.is_hybrid_swa_compress = getattr(
            model_config, "is_hybrid_swa_compress", False
        )
        self.use_mla_backend = self.model_config.attention_arch == AttentionArch.MLA
        self.attention_chunk_size = model_config.attention_chunk_size
        rope_scaling = getattr(
            model_config.hf_text_config, "rope_parameters", None
        ) or getattr(model_config.hf_text_config, "rope_scaling", {})
        self.model_is_mrope = (
            rope_scaling is not None and "mrope_section" in rope_scaling
        )
        self.enable_elastic_ep = server_args.elastic_ep_backend is not None
        self.forward_pass_id = 0
        self.init_new_workspace = False
        self.draft_model_idx = draft_model_idx
        self.enable_hisparse = server_args.enable_hisparse

        self.remote_instance_transfer_engine = None
        self.remote_instance_transfer_engine_session_id = ""
        self.remote_instance_transfer_engine_weight_info = None

        self.msprobe_debugger = None
        if server_args.msprobe_dump_config is not None:
            self.init_msprobe()

        # auxiliary hidden capture mode. TODO: expose this to server args?
        self.eagle_use_aux_hidden_state = False
        self.dflash_use_aux_hidden_state = False
        self.dflash_target_layer_ids = None
        self.dflash_draft_num_layers = None
        if self.spec_algorithm.is_eagle3() and not self.is_draft_worker:
            # load draft config
            draft_model_config = ModelConfig.from_server_args(
                server_args,
                model_path=(server_args.speculative_draft_model_path),
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            self.eagle_use_aux_hidden_state = True

            try:
                # get the aux layer from draft model config
                eagle_config = getattr(
                    draft_model_config.hf_config, "eagle_config", None
                )
                self.eagle_use_aux_hidden_state = eagle_config.get(
                    "use_aux_hidden_state", True
                )
                self.eagle_aux_hidden_state_layer_ids = eagle_config[
                    "eagle_aux_hidden_state_layer_ids"
                ]
            except:
                # if there is no aux layer, set to None
                self.eagle_aux_hidden_state_layer_ids = None

        if self.spec_algorithm.is_dflash() and not self.is_draft_worker:
            from sglang.srt.speculative.dflash_utils import (
                parse_dflash_draft_config,
            )

            # Select target layers to capture for building DFlash context features.
            draft_model_config = ModelConfig.from_server_args(
                server_args,
                model_path=(server_args.speculative_draft_model_path),
                model_revision=server_args.speculative_draft_model_revision,
                is_draft_model=True,
            )
            dflash_draft_config = parse_dflash_draft_config(
                draft_hf_config=draft_model_config.hf_config
            )
            draft_num_layers = dflash_draft_config.require_num_layers()
            trained_target_layers = dflash_draft_config.num_target_layers

            target_num_layers = getattr(
                self.model_config.hf_text_config, "num_hidden_layers", None
            )
            if target_num_layers is None:
                raise ValueError(
                    "DFLASH requires target num_hidden_layers in config. "
                    f"Got target={target_num_layers}."
                )
            target_num_layers = int(target_num_layers)

            if (
                trained_target_layers is not None
                and trained_target_layers != target_num_layers
            ):
                logger.warning(
                    "DFLASH draft config num_target_layers=%s differs from runtime target num_hidden_layers=%s; "
                    "selecting capture layers based on the runtime target model.",
                    trained_target_layers,
                    target_num_layers,
                )

            self.dflash_use_aux_hidden_state = True
            self.dflash_draft_num_layers = int(draft_num_layers)
            self.dflash_target_layer_ids = dflash_draft_config.resolve_target_layer_ids(
                target_num_layers=int(target_num_layers),
                draft_num_layers=int(draft_num_layers),
            )

        # Apply the rank zero filter to logger
        if server_args.show_time_cost:
            enable_show_time_cost()

        # Model-specific adjustment
        self.model_specific_adjustment()

        # Set the global server_args in the scheduler process
        set_global_server_args_for_scheduler(server_args)
        global_server_args = get_global_server_args()

        # FIXME: hacky set `use_mla_backend`
        global_server_args.use_mla_backend = self.use_mla_backend

        # Init OpenMP threads binding for CPU
        if self.device == "cpu":
            self.init_threads_binding()

        # Get available memory before model loading
        # 先初始化分布式环境，再加载模型。
        # 这样可以在权重加载前获得通信组和显存基线，后续 KV cache 大小估算会用到这个基线。
        pre_model_load_memory = self.init_torch_distributed()

        # Initialize MooncakeTransferEngine
        self.init_shared_mooncake_transfer_engine()

        # Init forward stream for overlap schedule
        self.forward_stream = torch.get_device_module(self.device).Stream()

        # CPU offload
        set_offloader(create_offloader_from_server_args(server_args, dp_rank=dp_rank))

        self._weight_checker = WeightChecker(model_runner=self)

        if envs.SGLANG_DETECT_SLOW_RANK.get():
            slow_rank_detector.execute()

        # Init mindspore running environment when model impl is "mindspore"
        self.init_mindspore_runner()

        # Update deep gemm configure
        if deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM:
            deep_gemm_wrapper.update_deep_gemm_config(gpu_id, server_args)

        # For hisparse (must be set before initialize() so CUDA graph capture can see it)
        self.hisparse_coordinator = None

        self._linear_attn_registry_cache: Any = _UNSET

        # Initialize the model runner
        # 进入 ModelRunner 主初始化流水线。
        # 这一调用会继续完成模型加载、KV cache、attention backend、kernel warmup 和 graph capture。
        self.initialize(pre_model_load_memory)
        self.check_quantized_moe_compatibility()

        if (
            self.server_args.elastic_ep_backend is not None
            and self.server_args.elastic_ep_rejoin
        ):
            join_process_groups()
            broadcast_global_expert_location_metadata(
                src_rank=self._get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=True
                )
            )
            ElasticEPStateManager.instance().reset()

        if self.is_multimodal:
            sanity_check_mm_pad_shift_value(self.model_config.vocab_size)

        # Temporary cached values
        self.support_pp = (
            "pp_proxy_tensors" in inspect.signature(self.model.forward).parameters
        )

        if self.pp_size > 1:
            assert (
                self.support_pp
            ), "Pipeline Parallel is not compatible with this model."

        # For weight updates
        self._model_update_group = {}
        self._weights_send_group = {}

    # 初始化 msprobe 精度调试工具。
    # 如果安装了 msprobe，它可以在 forward 过程中 dump tensor，帮助定位 NPU/混合精度场景下的数值问题。
    def init_msprobe(self):
        # Init the msprobe
        try:
            from msprobe.pytorch import PrecisionDebugger, seed_all
        except ImportError:
            logger.warning(
                "Please install msprobe for tensor data dump: pip install mindstudio-probe --pre, "
                "see https://gitcode.com/Ascend/msprobe for details."
            )
            return
        seed_all(mode=True)
        self.msprobe_debugger = PrecisionDebugger(
            config_path=self.server_args.msprobe_dump_config
        )

    # 初始化 MindSpore 后端相关通信。
    # 当 model_impl 为 MindSpore 且设备为 NPU 时，这里会调用 MindSpore runner 建立分布式环境。
    def init_mindspore_runner(self):
        # Init the mindspore runner
        # for now, there is only some communication initialization work
        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE and _is_npu:
            from sglang.srt.model_executor.mindspore_runner import init_ms_distributed

            init_ms_distributed(
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                server_args=self.server_args,
                port=self.dist_port,
            )

    # ModelRunner 的主体初始化流水线。
    # 这个函数按顺序加载模型、准备 MoE、配置 KV cache dtype、初始化内存池、建立 attention backend、warmup kernel、捕获 graph。
    # 很多高性能路径必须在这里提前完成，否则真实请求到来时会承担昂贵的首次初始化成本。
    def initialize(self, pre_model_load_memory: float):
        server_args = self.server_args

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=self.server_args.enable_memory_saver
        )

        if self.server_args.remote_instance_weight_loader_use_transfer_engine():
            self.remote_instance_init_transfer_engine()

        if not self.is_draft_worker:
            set_global_expert_location_metadata(
                compute_initial_expert_location_metadata(
                    server_args=server_args,
                    model_config=self.model_config,
                    moe_ep_rank=self.moe_ep_rank,
                )
            )
            if self.tp_rank == 0 and envs.SGLANG_LOG_EXPERT_LOCATION_METADATA.get():
                logger.info(
                    f"Initial expert_location_metadata: {get_global_expert_location_metadata()}"
                )

            set_global_expert_distribution_recorder(
                ExpertDistributionRecorder.init_new(
                    server_args,
                    get_global_expert_location_metadata(),
                    rank=self.tp_rank,
                )
            )

        # Expert parallelism
        self.eplb_manager = (
            EPLBManager(self)
            if self.server_args.enable_eplb and (not self.is_draft_worker)
            else None
        )
        self.expert_location_updater = ExpertLocationUpdater()

        if self.server_args.elastic_ep_backend:
            ElasticEPStateManager.init(self.server_args)
        self._token_oracle_manager = install_token_oracle_from_env(
            server_args=server_args,
            vocab_size=self.model_config.vocab_size,
        )
        # Load the model
        self.sampler = create_sampler()
        # 开始加载模型权重。
        # load_model 内部会进入 memory saver 上下文，调用 loader，并在加载后记录 dtype、sliding window、量化和 debug 状态。
        self.load_model()
        # 准备 MoE top-k 路由。
        # 对 MoE 模型而言，专家选择和 token 分发是 forward 性能的关键路径之一。
        self._prepare_moe_topk()

        # Load the expert backup client
        self.expert_backup_client = (
            ExpertBackupClient(self.server_args, self)
            if (
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
            else None
        )

        if (
            self.server_args.remote_instance_weight_loader_use_transfer_engine()
            # ModelExpress owns TransferEngine memory registration and metadata
            # publishing for backend=modelexpress. Re-registering here would
            # overlap the same weight buffers.
            and self.server_args.remote_instance_weight_loader_backend
            != RemoteInstanceWeightLoaderBackend.MODELEXPRESS
            and self.remote_instance_transfer_engine is not None
            and self.remote_instance_transfer_engine_weight_info is None
        ):
            # Register memory and upstream the transfer engine info to the bootstrap server
            self.remote_instance_transfer_engine_weight_info = register_memory_region(
                self.model, self.remote_instance_transfer_engine
            )
            self._register_to_engine_info_bootstrap()

        # For MTP models like DeepSeek-V3 or GLM-4.5, the MTP layer(s) are used separately as draft
        # models for speculative decoding. In those cases, `num_nextn_predict_layers` is used to
        # determine the number of layers.
        model_has_mtp_layers = self.model_config.num_nextn_predict_layers is not None
        model_num_layers = (
            self.model_config.num_nextn_predict_layers
            if self.is_draft_worker and model_has_mtp_layers
            else max(
                self.model_config.num_hidden_layers,
                self.model_config.num_attention_layers,
            )
        )
        if self.model_config.hf_config.architectures[0] == "MiMoV2MTP":
            model_num_layers = 1
        elif self.model_config.hf_config.architectures[0] == "Step3p5MTP":
            model_num_layers = 1
        self.start_layer = getattr(self.model, "start_layer", 0)
        self.end_layer = getattr(self.model, "end_layer", model_num_layers)
        self.num_effective_layers = self.end_layer - self.start_layer

        self.adjust_hybrid_swa_layers_for_pp()

        # For LoopCoder models, each loop has its own layer_id, so we need to multiply by loop_num
        loop_num = getattr(self.model_config.hf_config, "loop_num", 1)
        if loop_num > 1:
            self.num_effective_layers = self.num_effective_layers * loop_num

        assert (
            (not model_has_mtp_layers)
            or (self.spec_algorithm.is_none())
            or (
                (not self.spec_algorithm.is_none())
                and (self.num_effective_layers == model_num_layers)
            )
        ), "PP is not compatible with MTP models."

        # Apply torchao quantization
        torchao_applied = getattr(self.model, "torchao_applied", False)
        # In layered loading, torchao may have been applied
        if not torchao_applied:
            apply_torchao_config_to_model(
                self.model, get_global_server_args().torchao_config
            )

        # Apply torch TP if the model supports it
        supports_torch_tp = getattr(self.model, "supports_torch_tp", False)
        if self.tp_size > 1 and supports_torch_tp:
            self.apply_torch_tp()

        # Init lora
        if server_args.enable_lora:
            self.init_lora_manager()
            if not server_args.disable_cuda_graph:
                # Phase 1 of LoRA CUDA graph init: pre-allocate large MoE
                # intermediate buffers before init_memory_pool() so memory
                # profiling accounts for them.  Phase 2 (dense LoRA batch
                # metadata) is handled in CudaGraphRunner.__init__() via
                # lora_manager.init_cuda_graph_batch_info().
                self._init_lora_cuda_graph_moe_buffers()

        # Enable batch invariant mode
        if server_args.enable_deterministic_inference:
            from sglang.srt.batch_invariant_ops import enable_batch_invariant_mode

            enable_batch_invariant_mode()

        # Deduce KV cache dtype
        # 确定 KV cache dtype。
        # 注意它可能不同于模型权重 dtype，例如 FP8 KV cache 可以显著降低显存占用。
        self.configure_kv_cache_dtype()

        # Init memory pool and attention backends
        # 初始化 KV cache 与请求/token 内存池。
        # continuous batching 的核心是复用这些池中的位置，而不是为每个请求重新分配大块显存。
        self.init_memory_pool(pre_model_load_memory)

        # Must be called AFTER init_memory_pool so the pool object exists for
        # canary to monkey-patch, and BEFORE init_device_graphs so warmup
        # forwards captured into the graph see the patched pool methods.
        self.canary_manager = install_canary(
            server_args=server_args,
            model_runner=self,
            token_oracle_manager=self._token_oracle_manager,
        )

        # Init ngram embedding token table
        # 如果模型需要 ngram embedding，这里建立请求维度的 token table。
        # 该 table 会在 sampling 后更新，下一轮 forward 可以使用最新 token 历史。
        self.maybe_init_ngram_embedding()

        # Init routed experts capturer
        self.init_routed_experts_capturer()

        self.init_indexer_capturer()

        # TODO: Refactor device-specific init branches into platform interface (separate PR).
        # Must be called BEFORE init_device_graphs() so CUDA graph capture
        # runs with aux hidden state capture enabled.
        self.init_aux_hidden_state_capture()

        if self.device == "cuda" or self.device == "musa":
            self.init_cublas()
            if self.enable_hisparse:
                from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator
                from sglang.srt.mem_cache.sparsity import parse_hisparse_config

                hisparse_cfg = parse_hisparse_config(self.server_args)
                hisparse_top_k = getattr(
                    self.model_config.hf_text_config, "index_topk", hisparse_cfg.top_k
                )
                self.hisparse_coordinator = HiSparseCoordinator(
                    req_to_token_pool=self.req_to_token_pool,
                    token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                    top_k=hisparse_top_k,
                    device_buffer_size=hisparse_cfg.device_buffer_size,
                    device=self.device,
                    tp_group=(
                        self.attention_tp_group.cpu_group
                        if self.server_args.enable_dp_attention
                        else self.tp_group.cpu_group
                    ),
                    host_to_device_ratio=hisparse_cfg.host_to_device_ratio,
                )
            # 初始化 attention backend。
            # backend 会在每次 forward 前根据 ForwardBatch 准备 page table、seq lens、workspace 等 metadata，并最终调用具体 attention kernel。
            self.init_attention_backend()
            # 预热或调优 kernel。
            # 这一步把编译、autotune、workspace 初始化等开销尽量放到服务启动时完成。
            self.kernel_warmup()
            self._pre_initialize_flashinfer_allreduce_workspace()
            # 捕获 device graph。
            # decode 场景形状相对稳定，捕获后可以通过 replay 减少 Python 调度和 kernel launch 开销。
            self.init_device_graphs()
        elif self.device == "cpu":
            self.init_attention_backend()
            self.init_device_graphs()
        elif self.device == "npu":
            self.init_attention_backend()
            # lazy init for zbal with mix mode(before graph capture when enable_cuda_graph)
            if envs.SGLANG_ZBAL_LOCAL_MEM_SIZE.get() > 0 and not self.is_draft_worker:
                from sglang.srt.hardware_backend.npu.utils import lazy_init_zbal_gva_mem

                lazy_init_zbal_gva_mem(
                    self.device,
                    self.gpu_id,
                    get_world_group().rank_in_group,
                    get_world_group().world_size,
                    get_world_group().cpu_group,
                )
            self.init_device_graphs()
        elif current_platform.is_out_of_tree():
            self.init_attention_backend()
            if current_platform.support_cuda_graph():
                self.init_device_graphs()
            else:
                self.graph_runner = None
                self.graph_mem_usage = 0
        else:
            self.graph_runner = None
            self.graph_mem_usage = 0
            self.init_attention_backend()

        if server_args.forward_hooks:
            register_forward_hooks(self.model, server_args.forward_hooks)

        # Initialize piecewise CUDA graph
        # 初始化 piecewise CUDA graph。
        # 它不是捕获整个 forward，而是捕获 attention/MoE 等局部片段，以适配更动态的执行形状。
        self.init_piecewise_cuda_graphs()

        self.prealloc_symmetric_memory_pool()

        if self.canary_manager is not None and not self.is_draft_worker:
            self.canary_manager.mark_init_finished()

    # 在 pipeline parallel 场景下修正 hybrid sliding-window attention 的层信息。
    # PP 会把模型层切给不同 rank，因此原本的全局层配置需要映射到当前 rank 负责的局部层范围。
    def adjust_hybrid_swa_layers_for_pp(self):
        if not self.is_hybrid_swa:
            return

        if self.model_config.is_deepseek_v4_arch:
            return

        full_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "full_attention_layer_ids")
            and layer_idx in self.model_config.full_attention_layer_ids
        ]
        swa_attention_layer_ids = [
            layer_idx
            for layer_idx in range(self.start_layer, self.end_layer + 1)
            if hasattr(self.model_config, "swa_attention_layer_ids")
            and layer_idx in self.model_config.swa_attention_layer_ids
        ]
        self.model_config.swa_attention_layer_ids = swa_attention_layer_ids
        self.model_config.full_attention_layer_ids = full_attention_layer_ids

    # 初始化 routed experts 捕获器。
    # 它用于记录 MoE 路由时 token 被分配到哪些 expert，方便调试、负载均衡或统计专家分布。
    def init_routed_experts_capturer(self):
        if not self.server_args.disable_shared_experts_fusion and hasattr(
            self.model, "num_fused_shared_experts"
        ):
            num_fused_shared_experts = self.model.num_fused_shared_experts
        else:
            num_fused_shared_experts = 0

        set_global_experts_capturer(
            RoutedExpertsCapturer.create(
                enable=get_global_server_args().enable_return_routed_experts,
                model_config=self.model_config,
                num_fused_shared_experts=num_fused_shared_experts,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    # 初始化 DSA/indexer 捕获器。
    # 某些稀疏或动态索引 kernel 会产生索引信息，这里注册捕获器以便后续检查和调试。
    def init_indexer_capturer(self):
        enable = get_global_server_args().enable_return_indexer_topk
        # Producer wiring is CUDA-only (Indexer.forward_cuda + MLA skip_topk
        # path); other backends would create a capturer but never feed it.
        if enable and self.device != "cuda":
            logger.warning(
                "indexer-topk capture is CUDA-only; %s backend not yet wired. "
                "Disabling capturer.",
                self.device,
            )
            set_global_indexer_capturer(None)
            return

        hf_text_config = self.model_config.hf_text_config
        num_indexer_layers = get_num_indexer_layers(hf_text_config)
        index_topk = getattr(hf_text_config, "index_topk", 0)
        set_global_indexer_capturer(
            create_indexer_capturer(
                enable=enable,
                num_indexer_layers=num_indexer_layers,
                index_topk=index_topk,
                num_tokens=self.max_total_num_tokens + self.page_size,
                max_running_requests=self.max_running_requests,
                device=self.device,
            )
        )

    # 初始化辅助 hidden states 捕获逻辑。
    # EAGLE3、DFLASH 等 speculative 路径需要中间层 hidden states，target 模型会在这里安装对应捕获点。
    def init_aux_hidden_state_capture(self):
        """Configure auxiliary hidden state capture for speculative decoding.

        Must be called before CUDA graph capture so the captured graphs
        include aux hidden state output paths.
        """
        if self.eagle_use_aux_hidden_state:
            self.model.set_eagle3_layers_to_capture(
                self.eagle_aux_hidden_state_layer_ids
            )
        if self.dflash_use_aux_hidden_state:
            if not hasattr(self.model, "set_dflash_layers_to_capture"):
                raise ValueError(
                    f"Model {self.model.__class__.__name__} does not implement "
                    "set_dflash_layers_to_capture, which is required for DFLASH."
                )
            self.model.set_dflash_layers_to_capture(self.dflash_target_layer_ids)

    # 初始化远端实例权重传输引擎。
    # 当权重加载或同步需要跨实例传输时，这个 engine 负责底层数据通道和连接管理。
    def remote_instance_init_transfer_engine(self):
        try:
            from mooncake.engine import TransferEngine
        except ImportError as e:
            logger.warning(
                "Please install mooncake for using remote instance transfer engine: pip install mooncake"
            )
            return
        self.remote_instance_transfer_engine = TransferEngine()
        local_ip = get_local_ip_auto()
        self.remote_instance_transfer_engine.initialize(
            local_ip,
            "P2PHANDSHAKE",
            envs.MOONCAKE_PROTOCOL.get(),
            envs.MOONCAKE_DEVICE.get(),
        )
        self.remote_instance_transfer_engine_session_id = NetworkAddress(
            local_ip, self.remote_instance_transfer_engine.get_rpc_port()
        ).to_host_port_str()

    # 向传输引擎注册当前实例信息。
    # 远端实例需要知道本 rank 的地址、设备和通信元信息，才能建立正确的数据传输关系。
    def _register_to_engine_info_bootstrap(self):
        """Register transfer engine info with the EngineInfoBootstrapServer via HTTP PUT.

        The bootstrap server runs on node_rank==0. For multi-node setups, the
        host is derived from dist_init_addr. For single-node, use 127.0.0.1.
        """
        import requests as http_requests

        if self.server_args.dist_init_addr:
            # Multi-node: bootstrap server is on the head node (node_rank==0).
            # Derive host from dist_init_addr (shared across all nodes).
            bootstrap_host = (
                NetworkAddress.parse(self.server_args.dist_init_addr).resolved().host
            )
        else:
            bootstrap_host = "127.0.0.1"

        bootstrap_port = self.server_args.engine_info_bootstrap_port
        bootstrap_na = NetworkAddress(bootstrap_host, bootstrap_port)
        url = f"{bootstrap_na.to_url()}/register_transfer_engine_info"

        payload = {
            "tp_rank": self.tp_rank,
            "transfer_engine_info": {
                "session_id": self.remote_instance_transfer_engine_session_id,
                "weights_info_dict": self.remote_instance_transfer_engine_weight_info,
            },
        }

        try:
            resp = http_requests.put(url, json=payload, timeout=5)
            if resp.status_code == 200:
                logger.info(
                    f"Registered transfer engine info for tp_rank={self.tp_rank} "
                    f"with bootstrap server at {bootstrap_na}"
                )
            else:
                logger.error(
                    f"Failed to register transfer engine info for tp_rank={self.tp_rank}: "
                    f"{resp.status_code}, {resp.text}"
                )
        except Exception as e:
            logger.error(
                f"Failed to register transfer engine info for tp_rank={self.tp_rank}: {e}"
            )

    # 根据模型类型做运行时修正。
    # 不同模型家族可能需要特殊 rope、attention、linear attention 或配置兼容处理，这些差异在加载前统一收口。
    def model_specific_adjustment(self):
        server_args = self.server_args

        if self.is_multimodal:
            if not self.is_multimodal_chunked_prefill_supported:
                server_args.chunked_prefill_size = -1
                logger.info(
                    f"Automatically turn off --chunked-prefill-size as it is not supported for "
                    f"{self.model_config.hf_config.model_type}"
                )

        if (
            not self.use_mla_backend
            or server_args.attention_backend
            not in CHUNKED_PREFIX_CACHE_SUPPORTED_ATTENTION_BACKENDS
        ):
            server_args.disable_chunked_prefix_cache = True

        if not server_args.disable_chunked_prefix_cache:
            log_info_on_rank0(logger, "Chunked prefix cache is turned on.")

    # `check_quantized_moe_compatibility` 负责校验运行时假设。
    # 这类检查通常用来提前发现权重、并行配置或后端能力不匹配的问题。
    def check_quantized_moe_compatibility(self):
        if (
            quantization_config := getattr(
                self.model_config.hf_config, "quantization_config", None
            )
        ) is not None and (
            weight_block_size := quantization_config.get("weight_block_size", None)
        ) is not None:
            weight_block_size_n = weight_block_size[0]

            if self.tp_size % self.moe_ep_size != 0:
                raise ValueError(
                    f"tp_size {self.tp_size} must be divisible by ep_size {self.moe_ep_size}"
                )
            moe_tp_size = self.tp_size // self.moe_ep_size // self.moe_dp_size

            moe_intermediate_size = getattr(
                self.model_config.hf_text_config, "moe_intermediate_size", None
            )
            if moe_intermediate_size is None:
                return

            if moe_intermediate_size % moe_tp_size != 0:
                raise ValueError(
                    f"moe_intermediate_size {moe_intermediate_size} must be divisible by moe_tp_size ({moe_tp_size}) which is tp_size ({self.tp_size}) divided by moe_ep_size ({self.moe_ep_size})."
                )

            if (
                not envs.SGLANG_SHARED_EXPERT_TP1.get()
                and (moe_intermediate_size // moe_tp_size) % weight_block_size_n != 0
                and not _use_aiter
            ):
                raise ValueError(
                    f"For quantized MoE models, please make sure ({moe_intermediate_size=} / {moe_tp_size=}) % {weight_block_size_n=} == 0 "
                    f"where moe_tp_size is equal to tp_size ({self.tp_size}) divided by ep_size ({self.moe_ep_size}). "
                    f"You can fix this by setting arguments `--tp` and `--ep` correctly."
                )

    # 初始化 torch distributed 和 SGLang 的模型并行通信组。
    # 这里会设置当前 device，选择 backend，创建 TP/PP/DP/EP/attention DP 等 group。
    # 返回的 pre_model_load_memory 会用于后续估算模型权重和 KV cache 可用显存。
    def init_torch_distributed(self):
        tic = time.perf_counter()
        logger.info("Init torch distributed begin.")

        try:
            torch.get_device_module(self.device).set_device(self.gpu_id)
        except Exception:
            logger.warning(
                f"Context: {self.device=} {self.gpu_id=} {os.environ.get('CUDA_VISIBLE_DEVICES')=} {self.tp_rank=} {self.tp_size=}"
            )
            raise

        backend = get_default_distributed_backend(self.device)
        if self.device == "cuda" and self.server_args.elastic_ep_backend == "mooncake":
            backend = "mooncake"
            if self.server_args.mooncake_ib_device:
                from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                    get_ib_devices_for_gpu,
                )

                ib_device_for_gpu = get_ib_devices_for_gpu(
                    self.server_args.mooncake_ib_device, self.gpu_id
                )
                mooncake_ib_device = (
                    ib_device_for_gpu.split(",") if ib_device_for_gpu else []
                )
                try:
                    from mooncake import ep as mooncake_ep

                    mooncake_ep.set_device_filter(mooncake_ib_device)
                except:
                    pass  # A warning will be raised in `init_distributed_environment`

        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if not self.server_args.enable_p2p_check:
            monkey_patch_p2p_access_check()

        # Allow external orchestrators (e.g. trainpi) to override the distributed
        # init method.  When set to "env://", torch uses MASTER_ADDR/MASTER_PORT
        # env-vars and an externally-created TCPStore, completely avoiding port
        # conflicts with intra-host collocation.
        dist_init_method_override = envs.SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE.get()
        if dist_init_method_override:
            dist_init_method = dist_init_method_override
        elif self.server_args.dist_init_addr:
            na = NetworkAddress.parse(self.server_args.dist_init_addr)
            dist_init_method = na.to_tcp()
        else:
            dist_init_method = NetworkAddress(
                self.server_args.host or "127.0.0.1", self.dist_port
            ).to_tcp()
        set_custom_all_reduce(not self.server_args.disable_custom_all_reduce)
        set_mscclpp_all_reduce(self.server_args.enable_mscclpp)
        set_torch_symm_mem_all_reduce(self.server_args.enable_torch_symm_mem)

        if not self.is_draft_worker:
            if self.device == "cpu":
                if _is_cpu_amx_available or _is_cpu_arm64:
                    # Bind OpenMP threads to CPU cores
                    torch.ops.sgl_kernel.init_cpu_threads_env(self.local_omp_cpuid)

                    # Set local size to hint SGLang to use shared memory based AllReduce
                    os.environ["LOCAL_SIZE"] = str(self.tp_size)
                    torch.ops.sgl_kernel.initialize(self.tp_size, self.tp_rank)

                    @torch.library.register_fake("sgl_kernel::shm_allgather")
                    # 为 CPU shared-memory allgather 注册 fake kernel。
                    # torch compile/export 需要 fake implementation 推断输出形状；这里用 cat 模拟 TP rank 聚合后的结果。
                    def _(data, dim):
                        return torch.cat([data] * self.tp_size, dim=dim)

                else:
                    logger.warning(
                        "init_cpu_threads_env and shared memory based AllReduce is disabled, only intel amx backend and arm64 are supported"
                    )

            # Only initialize the distributed environment on the target model worker.
            init_distributed_environment(
                backend=backend,
                world_size=self.tp_size * self.pp_size,
                rank=self.tp_size * self.pp_rank + self.tp_rank,
                local_rank=self.gpu_id,
                distributed_init_method=dist_init_method,
                timeout=self.server_args.dist_timeout,
                moe_a2a_backend=self.server_args.moe_a2a_backend,
                recovered_rank=self.server_args.elastic_ep_rejoin,
            )
            initialize_model_parallel(
                tensor_model_parallel_size=self.tp_size,
                attention_data_parallel_size=self.dp_size,
                pipeline_model_parallel_size=self.pp_size,
                expert_model_parallel_size=self.moe_ep_size,
                attention_context_model_parallel_size=self.attn_cp_size,
                moe_data_model_parallel_size=self.moe_dp_size,
                duplicate_tp_group=self.server_args.enable_pdmux,
                enable_symm_mem=self.server_args.enable_symm_mem,
                recovered_rank=self.server_args.elastic_ep_rejoin,
            )
            initialize_dp_attention(
                server_args=self.server_args,
                model_config=self.model_config,
            )
            if is_npu():
                register_sgl_tp_rank(self.gpu_id)

            # Pre-warm NCCL/RCCL to eliminate cold-start latency in first request
            # Controlled by --pre-warm-nccl flag (default: enabled on AMD GPUs)
            if self.server_args.pre_warm_nccl and (
                self.tp_size > 1 or self.pp_size > 1 or self.moe_ep_size > 1
            ):
                warmup_start = time.perf_counter()
                tp_group_handle = get_tp_group().device_group

                # Single warmup all_reduce to initialize NCCL/RCCL communicator
                warmup_tensor = torch.zeros(1, device=torch.cuda.current_device())
                dist.all_reduce(warmup_tensor, group=tp_group_handle)
                current_platform.synchronize()

                warmup_elapsed = time.perf_counter() - warmup_start
                logger.info(
                    f"NCCL/RCCL warmup completed in {warmup_elapsed:.3f}s "
                    f"(tp_size={self.tp_size}, pp_size={self.pp_size}, ep_size={self.moe_ep_size})"
                )

        pre_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=get_world_group().world_size > 1,
            cpu_group=get_world_group().cpu_group,
        )
        # 保存 tensor parallel group。
        # 后续权重切分、all-reduce、attention/MLP gather-scatter 都会使用这个 group。
        self.tp_group = get_tp_group()
        # 读取 pipeline parallel 通信组。
        # 后续 forward_batch_generation 会用 pp_group.is_last_rank 判断当前 rank 是否负责采样。
        self.pp_group = get_pp_group()
        self.attention_tp_group = get_attention_tp_group()

        # Check memory for tensor parallelism
        local_gpu_memory = get_available_gpu_memory(self.device, self.gpu_id)
        if self.tp_size > 1 and not self.is_draft_worker:
            if pre_model_load_memory < local_gpu_memory * 0.9:
                msg = "The memory capacity is unbalanced. Some GPUs may be occupied by other processes. "
                msg += f"{pre_model_load_memory=}, {local_gpu_memory=}, {local_gpu_memory * 0.9=}"
                if envs.SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK.get():
                    raise RuntimeError(msg)
                else:
                    logger.warning(msg)

        logger.info(
            f"Init torch distributed ends. elapsed={time.perf_counter() - tic:.2f} s, "
            f"mem usage={(before_avail_memory - local_gpu_memory):.2f} GB"
        )
        return pre_model_load_memory

    # 初始化共享 Mooncake transfer engine。
    # Mooncake 用于某些 disaggregation/远端传输场景，这里准备跨组件共享的传输通道。
    def init_shared_mooncake_transfer_engine(self):
        """
        Need MooncakeTransferEngine when:
        1) PD disaggregation uses mooncake for KV transfer (prefill/decode)
        2) HiCache uses mooncake storage backend
        3) Encoder disaggregation uses mooncake
        """
        use_mooncake_te = (
            (
                self.server_args.disaggregation_mode != "null"
                and self.server_args.disaggregation_transfer_backend == "mooncake"
            )
            or (
                self.server_args.enable_hierarchical_cache
                and self.server_args.hicache_storage_backend == "mooncake"
                and envs.SGLANG_HICACHE_MOONCAKE_REUSE_TE.get()
            )
            or (
                self.server_args.encoder_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.language_only
                and self.server_args.encoder_transfer_backend == "mooncake"
            )
            or (
                self.server_args.enable_elastic_expert_backup
                and self.server_args.elastic_ep_backend is not None
            )
        )

        if use_mooncake_te:
            from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
                init_mooncake_transfer_engine,
            )

            init_mooncake_transfer_engine(
                hostname=get_local_ip_auto(),
                gpu_id=self.gpu_id,
                ib_device=(
                    self.server_args.disaggregation_ib_device
                    or self.server_args.mooncake_ib_device
                ),
            )

    # 加载模型权重并完成模型对象的运行时修正。
    # 这里会构造 LoadConfig/DeviceConfig，进入 memory saver 上下文，调用模型 loader，处理远端权重、量化、滑窗、dtype、debug hook 等状态。
    # 加载完成后，ModelRunner 才能根据模型结构建立 KV cache 与 attention backend。
    def load_model(self):
        tic_total = time.perf_counter()
        before_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Load weight begin. avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        # This can reduce thread conflicts and speed up weight loading.
        if self.device != "cpu":
            torch.set_num_threads(1)
        if self.device == "cuda":
            if torch.cuda.get_device_capability()[0] < 8:
                logger.info(
                    "Compute capability below sm80. Use float16 due to lack of bfloat16 support."
                )
                self.server_args.dtype = "float16"
                self.model_config.dtype = torch.float16
                if torch.cuda.get_device_capability()[1] < 5:
                    raise RuntimeError("SGLang only supports sm75 and above.")

        set_cuda_arch()

        # Prepare the model config
        from sglang.srt.configs.modelopt_config import ModelOptConfig

        modelopt_config = ModelOptConfig(
            quant=self.server_args.modelopt_quant,
            checkpoint_restore_path=self.server_args.modelopt_checkpoint_restore_path,
            checkpoint_save_path=self.server_args.modelopt_checkpoint_save_path,
            export_path=self.server_args.modelopt_export_path,
            quantize_and_serve=self.server_args.quantize_and_serve,
        )

        self.load_config = LoadConfig(
            load_format=self.server_args.load_format,
            download_dir=self.server_args.download_dir,
            model_loader_extra_config=self.server_args.model_loader_extra_config,
            tp_rank=self.tp_rank,
            remote_instance_weight_loader_seed_instance_ip=self.server_args.remote_instance_weight_loader_seed_instance_ip,
            remote_instance_weight_loader_seed_instance_service_port=self.server_args.remote_instance_weight_loader_seed_instance_service_port,
            remote_instance_weight_loader_send_weights_group_ports=self.server_args.remote_instance_weight_loader_send_weights_group_ports,
            remote_instance_weight_loader_backend=self.server_args.remote_instance_weight_loader_backend,
            remote_instance_weight_loader_transfer_engine=self.remote_instance_transfer_engine,
            remote_instance_weight_loader_transfer_engine_session_id=self.remote_instance_transfer_engine_session_id,
            modelexpress_url=self.server_args.modelexpress_url,
            modelexpress_transport=self.server_args.modelexpress_transport,
            modelopt_config=modelopt_config,
            rl_quant_profile=self.server_args.rl_quant_profile,
            draft_model_idx=self.draft_model_idx,
        )
        if self.device == "cpu":
            self.model_config = adjust_config_with_unaligned_cpu_tp(
                self.model_config, self.load_config, self.tp_size
            )

        if (
            self.server_args.load_format == LoadFormat.REMOTE_INSTANCE
            and self.server_args.remote_instance_weight_loader_backend
            == RemoteInstanceWeightLoaderBackend.NCCL
        ):
            if self.tp_rank == 0:
                instance_ip = NetworkAddress.resolve_host(socket.gethostname())
                t = threading.Thread(
                    target=trigger_init_weights_send_group_for_remote_instance_request,
                    args=(
                        self.server_args.remote_instance_weight_loader_seed_instance_ip,
                        self.server_args.remote_instance_weight_loader_seed_instance_service_port,
                        self.server_args.remote_instance_weight_loader_send_weights_group_ports,
                        instance_ip,
                    ),
                )
                t.start()

        # Load the model
        # Remove monkey_patch when linear.py quant remove dependencies with vllm
        monkey_patch_vllm_parallel_state()

        enable_cpu_backup = self.server_args.enable_weights_cpu_backup or (
            self.is_draft_worker and self.server_args.enable_draft_weights_cpu_backup
        )
        with self.memory_saver_adapter.region(
            GPU_MEMORY_TYPE_WEIGHTS,
            enable_cpu_backup=enable_cpu_backup,
        ):
            self.loader = get_model_loader(
                load_config=self.load_config,
                model_config=self.model_config,
            )
            self.model = self.loader.load_model(
                model_config=self.model_config,
                device_config=DeviceConfig(self.device, self.gpu_id),
            )
            if hasattr(self.loader, "remote_instance_transfer_engine_weight_info"):
                self.remote_instance_transfer_engine_weight_info = (
                    self.loader.remote_instance_transfer_engine_weight_info
                )
        # Cache needs to be cleared after loading model weights (in the self.loader.load_model function).
        # To avoid conflict with memory_saver_adapter.region, empty_cache operation is now moved here.
        if _is_npu:
            torch.npu.empty_cache()
        monkey_patch_vllm_parallel_state(reverse=True)

        if not self.is_draft_worker:
            get_offloader().post_init()

        # Register model for layerwise NVTX profiling if enabled
        if self.server_args.enable_layerwise_nvtx_marker:
            pyt_hooks = PytHooks()
            pyt_hooks.register_hooks(self.model, module_prefix="model")

        if self.server_args.kv_cache_dtype == "fp8_e4m3":
            if self.server_args.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    self.model.load_kv_cache_scales(
                        self.server_args.quantization_param_path
                    )
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        self.server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__,
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        # Parse other args
        self.sliding_window_size = None
        if hasattr(self.model, "get_attention_sliding_window_size"):
            self.sliding_window_size = self.model.get_attention_sliding_window_size()
        elif (
            self.model_config.is_hybrid_swa
            and self.model_config.sliding_window_size is not None
        ):
            # sliding window field in model config may have different meaning for different kinds of models (e.g., dllm), here we only consider the sliding window in SWA model
            self.sliding_window_size = self.model_config.sliding_window_size
        elif self.model_config.attention_chunk_size is not None:
            self.sliding_window_size = self.model_config.attention_chunk_size
            logger.info(
                f"Setting sliding_window_size to be attention_chunk_size: {self.sliding_window_size}"
            )

        self.dtype = self.model_config.dtype

        after_avail_memory = get_available_gpu_memory(self.device, self.gpu_id)
        self.weight_load_mem_usage = before_avail_memory - after_avail_memory
        # Get quantization config from ModelConfig
        # This handles both config.json (standard) and hf_quant_config.json (ModelOpt)
        quant_str = self.model_config.get_quantization_config_log_str()

        logger.info(
            f"Load weight end. "
            f"elapsed={time.perf_counter() - tic_total:.2f} s, "
            f"type={type(self.model).__name__}, "
            f"{quant_str + ', ' if quant_str else ''}"
            f"avail mem={after_avail_memory:.2f} GB, "
            f"mem usage={self.weight_load_mem_usage:.2f} GB."
        )
        if self.server_args.debug_tensor_dump_output_folder is not None:
            dump_folder = self.server_args.debug_tensor_dump_output_folder
            if self.spec_algorithm.is_eagle():
                role = "draft" if self.is_draft_worker else "target"
                dump_folder = os.path.join(dump_folder, role)
            register_forward_hook_for_model(
                self.model,
                dump_folder,
                self.server_args.debug_tensor_dump_layers,
                self.tp_size,
                self.tp_rank,
                self.pp_rank,
            )

        if dumper.may_enable:
            dumper.apply_source_patches()
            dumper.register_non_intrusive_dumper(self.model)

        # Pre-expand RoPE cache before CUDA Graph capture
        reserve_rope_cache_for_long_sequences(
            self.model,
            self.server_args,
            self.model_config,
            logger,
        )

        if self.server_args.elastic_ep_backend == "mooncake":
            # Mooncake does not support `monitored_barrier`
            dist.barrier(group=get_tp_group().cpu_group)
        else:
            # Handle the case where some ranks do not finish loading.
            try:
                dist.monitored_barrier(
                    group=get_tp_group().cpu_group,
                    timeout=datetime.timedelta(
                        seconds=UNBALANCED_MODEL_LOADING_TIMEOUT_S
                    ),
                    wait_all_ranks=True,
                )
            except RuntimeError:
                raise ValueError(
                    f"TP rank {self.tp_rank} could finish the model loading, but there are other ranks that didn't finish loading. It is likely due to unexpected failures (e.g., OOM) or a slow node."
                ) from None

    # MoE 模型的路由通常依赖 top-k expert 选择。
    # 这个函数根据当前 MoE backend 做兼容性准备，确保后续 forward 中路由 kernel 能拿到正确配置。
    def _prepare_moe_topk(self):
        balancer_cls = None
        num_prepared = 0
        num_routed_experts = None
        for module in self.model.modules():
            if not isinstance(module, (TopK, HashTopK)):
                continue
            if (
                not module.enable_deepep_waterfill
                or module.deepep_waterfill_balancer is not None
            ):
                continue
            if num_routed_experts is None:
                num_routed_experts = getattr(
                    self.model_config.hf_config, "n_routed_experts", None
                )
                if num_routed_experts is None:
                    raise ValueError(
                        "DeepEP waterfill requires model config n_routed_experts."
                    )
            if balancer_cls is None:
                from sglang.srt.layers.moe.deepep_waterfill import (
                    DeepEPWaterfillBalancer,
                )

                balancer_cls = DeepEPWaterfillBalancer
            # Static EPLB remaps TopK ids to physical expert ids before Waterfill.
            # Redundant experts therefore need to be included in the per-rank
            # expert count used for Waterfill's shared-expert slot remapping.
            num_physical_routed_experts = (
                num_routed_experts + self.server_args.ep_num_redundant_experts
            )
            if isinstance(module, TopK):
                routed_scaling_factor = module.topk_config.routed_scaling_factor
            else:
                routed_scaling_factor = module.routed_scaling_factor
            module.deepep_waterfill_balancer = balancer_cls(
                num_routed_experts=num_physical_routed_experts,
                world_size=self.moe_ep_size,
                rank=self.moe_ep_rank,
                layer_id=module.layer_id,
                routed_scaling_factor=(
                    routed_scaling_factor if routed_scaling_factor is not None else 1.0
                ),
            )
            num_prepared += 1
        if num_prepared:
            log_info_on_rank0(
                logger, f"Prepared {num_prepared} DeepEP waterfill TopK modules."
            )

    # 更新 MoE expert 在各 rank 上的位置映射。
    # EPLB 或 elastic expert parallel 会动态调整 expert 布局，forward 前必须让路由逻辑看到最新位置。
    def update_expert_location(
        self,
        new_expert_location_metadata: ExpertLocationMetadata,
        update_layer_ids: List[int],
    ):
        p2p_missing_logical_experts = self.expert_location_updater.update(
            self.model.routed_experts_weights_of_layer,
            new_expert_location_metadata,
            update_layer_ids=update_layer_ids,
            nnodes=self.server_args.nnodes,
            rank=self.tp_rank,
        )

        if len(p2p_missing_logical_experts) > 0:
            # Load the missing expert weights from disk
            if callable(getattr(self.model, "generate_weight_name_filter", None)):
                # Filter and load only missing expert weights
                weight_name_filter = self.model.generate_weight_name_filter(
                    p2p_missing_logical_experts
                )
            else:
                # Do a full reload from disk/DRAM
                logger.info(
                    "[Elastic EP] Model does not implement generate_weight_name_filter. "
                    "Performing full weight reload."
                )
                weight_name_filter = None

            if (
                self.expert_backup_client is not None
                and self.expert_backup_client.use_backup
            ):
                # Load the missing weights from the DRAM backup
                self.expert_backup_client.update_weights(weight_name_filter)
            else:
                # Load the missing weights from disk
                self.update_weights_from_disk(
                    get_global_server_args().model_path,
                    get_global_server_args().load_format,
                    weight_name_filter=weight_name_filter,
                )

    # `maybe_recover_ep_ranks` 是可选特性的懒初始化或条件更新入口。
    # 只有模型结构或 server_args 启用对应能力时，这段逻辑才会产生实际效果。
    def maybe_recover_ep_ranks(self):
        # TODO(perf): `active_ranks.all()` on a CUDA tensor triggers host-device
        # synchronization, and this function is on the forward-path.
        # This check only runs when `--elastic-ep-backend` is enabled, so the
        # synchronization overhead does not propagate to other configs.
        # Leave for future optimization of the elastic EP path.
        if self.tp_group.active_ranks.all() and self.tp_group.active_ranks_cpu.all():
            return

        tp_active_ranks = self.tp_group.active_ranks.detach().cpu().numpy()
        tp_active_ranks_cpu = self.tp_group.active_ranks_cpu.detach().numpy()
        tp_active_ranks &= tp_active_ranks_cpu
        # NOTE: `ranks_to_recover` uses indices in `tp_group`. For the current
        # Mooncake elastic EP implementation we assume `--pp-size=1`, so the
        # tp-group index is the same as the global rank index.
        ranks_to_recover = [
            i for i in range(len(tp_active_ranks)) if not tp_active_ranks[i]
        ]

        # try_recover_ranks polls peer state via Mooncake EP backend.
        # Mooncake's internal semantics guarantee that all ranks observe
        # consistent peer readiness state, so collective operations below
        # are safe even though polling appears local.
        if ranks_to_recover and try_recover_ranks(ranks_to_recover):
            self.forward_pass_id = 0
            self.eplb_manager.reset_generator()
            broadcast_global_expert_location_metadata(
                src_rank=self._get_healthy_expert_location_src_rank(
                    invoked_in_elastic_ep_rejoin_path=False
                )
            )
            ElasticEPStateManager.instance().reset()

            broadcast_pyobj(
                [self.server_args.random_seed],
                get_world_group().rank,
                get_world_group().cpu_group,
                src=get_world_group().ranks[0],
            )
            logger.info(f"recover ranks {ranks_to_recover} done")

    # 选择可用于恢复 expert location 的健康源 rank。
    # 恢复时需要从仍然持有可信 expert 元数据的 rank 复制状态。
    def _get_healthy_expert_location_src_rank(
        self, invoked_in_elastic_ep_rejoin_path: bool
    ) -> int:
        world_group = get_world_group()
        # NOTE: do not key off `self.server_args.elastic_ep_rejoin` here.
        # A rank that was started as a rejoin rank may later act as a healthy
        # rank in a subsequent recovery cycle.
        local_rejoin_flag = bool(invoked_in_elastic_ep_rejoin_path)
        gathered_rejoin_flags = world_group.all_gather_object(local_rejoin_flag)

        for rank_in_group, is_rejoin_rank in enumerate(gathered_rejoin_flags):
            if not is_rejoin_rank:
                return world_group.ranks[rank_in_group]

        raise RuntimeError(
            "No healthy rank found for broadcasting expert location metadata. "
            "All ranks are marked as elastic_ep_rejoin."
        )

    # 从磁盘路径加载新的权重并更新当前模型。
    # worker 层负责接收请求对象，ModelRunner 层负责真正调用 loader、同步 rank 并刷新模型状态。
    # 这类接口常用于不停服替换权重或把模型切换到新的 checkpoint。
    def update_weights_from_disk(
        self,
        model_path: str,
        load_format: str,
        weight_name_filter: Optional[Callable[[str], bool]] = None,
        recapture_cuda_graph: bool = False,
    ) -> tuple[bool, str]:
        """Update engine weights in-place from the disk."""
        logger.info(
            f"Update engine weights online from disk begin. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id, empty_cache=False):.2f} GB"
        )

        target_device = torch.device(self.device)
        self.model_config.model_path = model_path
        load_config = LoadConfig(load_format=load_format)

        # Only support DefaultModelLoader for now
        loader = get_model_loader(load_config, self.model_config)
        if not isinstance(loader, DefaultModelLoader):
            message = f"Failed to get model loader: {loader}."
            return False, message

        # 构造权重迭代器。
        # loader 会从当前 ModelConfig 和模型对象中找到权重来源；如果传入 weight_name_filter，则只保留需要更新的参数。
        def get_weight_iter(config):
            iter = loader._get_weights_iterator(
                DefaultModelLoader.Source.init_new(config, self.model)
            )
            if weight_name_filter is not None:
                iter = (
                    (name, weight) for name, weight in iter if weight_name_filter(name)
                )

            return iter

        # 执行实际权重写入。
        # load_weights_and_postprocess 会把迭代器中的参数加载到模型，并完成 loader 需要的后处理。
        def model_load_weights(model, iter):
            loader.load_weights_and_postprocess(model, iter, target_device)
            return model

        with set_default_torch_dtype(self.model_config.dtype):
            try:
                iter = get_weight_iter(self.model_config)
            except Exception as e:
                message = f"Failed to get weights iterator: {e}."
                return False, message
            try:
                model = model_load_weights(self.model, iter)
            except Exception as e:
                message = (
                    f"Failed to update weights: {e}.\nRolling back to original weights."
                )
                del iter
                gc.collect()
                iter = get_weight_iter(self.model_config)
                self.model = model_load_weights(self.model, iter)
                return False, message

        self.model = model
        self.server_args.model_path = model_path
        self.server_args.load_format = load_format
        self.load_config = load_config

        if recapture_cuda_graph and (
            self.device == "cuda"
            or self.device == "musa"
            or (
                current_platform.is_out_of_tree()
                and current_platform.support_cuda_graph()
            )
        ):
            self.init_device_graphs()

        logger.info("Update weights end.")
        return True, "Succeeded to update model weights."

    # 初始化向远端实例发送权重所需的通信组。
    # 这个路径用于把当前实例的权重同步给另一个远端服务实例，常见于模型迁移、实例扩容或远端热更新。
    def init_weights_send_group_for_remote_instance(
        self,
        master_address,
        ports,
        group_rank,
        world_size,
        group_name,
        backend="nccl",
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        logger.info(
            f"init custom process group: tp_rank={self.tp_rank}, gpu_id={self.gpu_id}, master_address={master_address}, master_port={group_port}, "
            f"group_rank={group_rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        current_platform.empty_cache()
        success = False
        message = ""
        try:
            na = NetworkAddress(master_address, group_port)
            self._weights_send_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=group_rank,
                group_name=group_name,
                device_id=torch.device("cuda", self.gpu_id),
            )
            dist.barrier(group=self._weights_send_group[group_name])
            success = True
            message = f"Succeeded to init group through {na.to_host_port_str()} group."
        except Exception as e:
            message = f"Failed to init group: {e}."
            logger.error(message)

        current_platform.empty_cache()
        return success, message

    # 把当前模型权重发送到远端实例。
    # 函数会依赖前面建立的 send group，把每个 rank 持有的参数分片传到对应接收端。
    def send_weights_to_remote_instance(
        self,
        master_address,
        ports,
        group_name,
    ):
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        ports_list = ports.split(",")
        assert (
            len(ports_list) == self.tp_size
        ), f"Expected {self.tp_size} ports, but got {len(ports_list)} ports."
        group_port = ports_list[self.tp_rank]
        group_name = f"{group_name}_{group_port}_{self.tp_rank}"

        if self._weights_send_group[group_name] is not None:
            send_group = self._weights_send_group[group_name]
        else:
            message = f"Group {group_name} not in _weights_send_group list. Please call `init_weights_send_group_for_remote_instance` first."
            logger.error(message)
            return False, message

        current_platform.empty_cache()
        success = False
        na = NetworkAddress(master_address, group_port)
        message = ""
        try:
            for _, weights in self.model.named_parameters():
                torch.distributed.broadcast(
                    weights,
                    src=0,
                    group=send_group,
                )
            success = True
            message = f"Succeeded to send weights through {na.to_host_port_str()} {group_name}."
        except Exception as e:
            message = f"Failed to send weights: {e}."
            logger.error(message)

        # destroy the process group after sending weights
        del self._weights_send_group[group_name]
        torch.distributed.distributed_c10d.destroy_process_group(send_group)
        current_platform.empty_cache()
        return success, message

    # 初始化权重更新使用的分布式通信组。
    # 多 rank 模型在热更新时必须让每个 rank 收到自己负责的权重分片，因此需要单独的更新组来协调通信。
    def init_weights_update_group(
        self,
        master_address,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend="nccl",
    ):
        """Initialize the Torch process group for model parameter updates.

        `_model_update_group` is used in the RLHF workflow, where rank
        0 is the actor model in the training engine, and the other ranks are
        the inference engine, which is used for rollout.

        In the RLHF workflow, the training engine updates the model
        weights/parameters online, and broadcasts them to the inference
        engine through the `_model_update_group` process group.
        """
        assert (
            torch.distributed.is_initialized()
        ), "Default torch process group must be initialized"
        assert group_name != "", "Group name cannot be empty"

        rank = rank_offset + self.tp_rank

        logger.info(
            f"init custom process group: master_address={master_address}, master_port={master_port}, "
            f"rank_offset={rank_offset}, rank={rank}, world_size={world_size}, group_name={group_name}, backend={backend}"
        )

        try:
            na = NetworkAddress(master_address, master_port)
            self._model_update_group[group_name] = init_custom_process_group(
                backend=backend,
                init_method=na.to_tcp(),
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
            return True, "Succeeded to initialize custom process group."
        except Exception as e:
            message = f"Failed to initialize custom process group: {e}."
            logger.error(message)
            return False, message

    # 销毁权重更新通信组。
    # 当一次热更新流程结束或不再需要该 group 时释放资源，避免长期占用分布式通信句柄。
    def destroy_weights_update_group(self, group_name):
        try:
            if group_name in self._model_update_group:
                pg = self._model_update_group.pop(group_name)
                torch.distributed.destroy_process_group(pg)
                return True, "Succeeded to destroy custom process group."
            else:
                return False, "The group to be destroyed does not exist."
        except Exception as e:
            message = f"Failed to destroy custom process group: {e}."
            logger.error(message)
            return False, message

    # 通过分布式通信接收并更新权重。
    # 这条路径适合权重已经分散在多个 rank 或远端发送端的场景，避免把完整权重先聚合到单个进程。
    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        load_format: Optional[str] = None,
    ):
        """
        Update specific parameter in the model weights online
        through `_model_update_group` process group.

        Args:
            name: the name of the parameter to be updated.
            dtype: the data type of the parameter to be updated.
            shape: the shape of the parameter to be updated.
        """

        assert group_name in self._model_update_group, (
            f"Group {group_name} not in {list(self._model_update_group.keys())}. "
            "Please call `init_weights_update_group` first."
        )

        if load_format == "flattened_bucket":
            return self._update_bucketed_weights_from_distributed(
                names, dtypes, shapes, group_name
            )
        try:
            weights = []
            handles = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                weight = torch.empty(shape, dtype=target_dtype, device=self.device)
                handles.append(
                    torch.distributed.broadcast(
                        weight,
                        src=0,
                        group=self._model_update_group[group_name],
                        async_op=True,
                    )
                )
                weights.append((name, weight))
            for handle in handles:
                handle.wait()

            self.model.load_weights(weights)
            return True, "Succeeded to update parameter online."

        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    # 处理 bucket 化后的分布式权重更新。
    # bucket 把多个小 tensor 合并传输，可以减少通信次数；这里负责拆解 bucket 并把参数写回模型。
    def _update_bucketed_weights_from_distributed(
        self, names, dtypes, shapes, group_name
    ):
        try:
            named_tensors = []
            for name, dtype, shape in zip(names, dtypes, shapes):
                target_dtype = (
                    dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
                )
                named_tensors.append(
                    (name, torch.empty(shape, dtype=target_dtype, device=self.device))
                )
            bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            flattened_tensor = bucket.get_flattened_tensor()
            torch.distributed.broadcast(
                flattened_tensor,
                src=0,
                group=self._model_update_group[group_name],
            )
            reconstructed_tensors = bucket.reconstruct_tensors()
            self.model.load_weights(reconstructed_tensors)
            return True, f"Succeeded to update parameter online."
        except Exception as e:
            error_msg = (
                f"Failed to update parameter online: {e}. "
                f"The full weights of the ModelRunner are partially updated. "
                f"Please discard the whole weights."
            )
            logger.error(error_msg)
            return False, error_msg

    # 从内存中的 tensor 字典更新权重。
    # 这条路径绕过磁盘文件，适合控制面已经把参数 tensor 直接传给 worker 的情况。
    def update_weights_from_tensor(
        self,
        named_tensors: List[Tuple[str, Union[torch.Tensor, "LocalSerializedTensor"]]],
        load_format: Optional[str] = None,
    ):
        monkey_patch_torch_reductions()
        if load_format == "flattened_bucket":
            # Handle flattened bucket format
            return self._update_weights_from_flattened_bucket(
                flattened_tensor_bucket_dict=named_tensors
            )

        # We need to get device after patch otherwise the device would be wrong
        device_module = torch.get_device_module(self.device)
        infered_device = device_module.current_device()

        named_tensors = [
            (name, _unwrap_tensor(tensor, tp_rank=self.tp_rank, device=infered_device))
            for name, tensor in named_tensors
        ]
        if load_format == "direct":
            _model_load_weights_direct(self.model, named_tensors)
        elif load_format in self.server_args.custom_weight_loader:
            custom_loader = dynamic_import(load_format)
            custom_loader(self.model, named_tensors)
        elif load_format is None:
            self.model.load_weights(named_tensors)
        else:
            raise NotImplementedError(f"Unknown load_format={load_format}")
        return True, "Success"

    # 处理 flattened bucket 格式的权重更新。
    # 多个参数被打平成一个连续 tensor 传输；这里根据 metadata 还原每个参数并写入模型。
    def _update_weights_from_flattened_bucket(
        self,
        flattened_tensor_bucket_dict,
    ):
        """Handle flattened bucket format for weight updates"""
        flattened_tensor = flattened_tensor_bucket_dict["flattened_tensor"]
        metadata = flattened_tensor_bucket_dict["metadata"]

        # Convert metadata dict to our format
        converted_metadata = []
        for meta in metadata:
            converted_meta = FlattenedTensorMetadata(
                name=meta.name,
                shape=meta.shape,
                dtype=meta.dtype,
                start_idx=meta.start_idx,
                end_idx=meta.end_idx,
                numel=meta.numel,
            )
            converted_metadata.append(converted_meta)

        # Create bucket and reconstruct tensors
        bucket = FlattenedTensorBucket(
            flattened_tensor=flattened_tensor, metadata=converted_metadata
        )
        reconstructed_tensors = bucket.reconstruct_tensors()

        # Load the reconstructed tensors using the standard method
        self.model.load_weights(reconstructed_tensors)

        return True, "Success"

    # 按参数名读取当前模型权重。
    # 常用于热更新前后的校验、debug，或把本实例的某些权重片段发送给远端实例。
    def get_weights_by_name(
        self, name: str, truncate_size: int = 100
    ) -> Optional[torch.Tensor]:
        """Get the weights of the parameter by its name. Similar to `get_parameter` in Hugging Face.

        Only used for unit test with an unoptimized performance.
        For optimized performance, please use torch.save and torch.load.
        """
        # TODO: (chenyang) Add support for Qwen models.
        try:
            return self.model.get_weights_by_name(
                name, truncate_size, tp_size=self.tp_size
            )
        except Exception as e:
            logger.error(f"Error when getting parameter {name}: {e}")
            return None

    # 初始化 LoRA manager。
    # 该对象负责 adapter 的加载、卸载、batch 级选择以及 LoRA 权重在 TP rank 上的切分和缓存。
    def init_lora_manager(self):
        self.lora_manager = LoRAManager(
            base_model=self.model,
            base_hf_config=self.model_config.hf_config,
            max_loras_per_batch=self.server_args.max_loras_per_batch,
            load_config=self.load_config,
            dtype=self.dtype,
            server_args=self.server_args,
            lora_backend=self.server_args.lora_backend,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            max_lora_rank=self.server_args.max_lora_rank,
            target_modules=self.server_args.lora_target_modules,
            lora_paths=self.server_args.lora_paths,
        )

    # 为 LoRA 与 CUDA graph/MoE 组合场景准备额外 buffer。
    # graph capture 对内存地址稳定性要求高，因此 LoRA/MoE 的动态状态需要提前放入固定 buffer。
    def _init_lora_cuda_graph_moe_buffers(self):
        """Phase 1 of LoRA CUDA graph init: pre-allocate MoE intermediate buffers.

        Must be called before init_memory_pool() so that memory profiling
        sees the reduced available memory and sizes KV cache correctly.
        All MoE LoRA layers share one set of buffers (managed by the
        lora_backend) since they execute sequentially during forward.

        Phase 2 (dense LoRA batch metadata) is handled later in
        CudaGraphRunner.__init__() via lora_manager.init_cuda_graph_batch_info(),
        because it needs capture-time parameters (max_bs, num_tokens_per_bs)
        that are only available at that stage.
        """
        from sglang.srt.lora.layers import FusedMoEWithLoRA

        max_bs = self.server_args.cuda_graph_max_bs
        max_loras = self.server_args.max_loras_per_batch
        for module in self.model.modules():
            if isinstance(module, FusedMoEWithLoRA):
                self.lora_manager.init_cuda_graph_moe_buffers(
                    max_bs, max_loras, self.dtype, module
                )
                logger.info(
                    f"Pre-allocated shared MoE LoRA CUDA graph buffers "
                    f"(max_bs={max_bs}, max_loras={max_loras})"
                )
                break

    # 动态加载一个 LoRA adapter。
    # ModelRunner 会把 LoRA 权重放入 LoRA manager 管理；后续请求可以通过 lora id 选择使用哪个 adapter。
    def load_lora_adapter(self, lora_ref: LoRARef):
        """Load a new lora adapter from disk or huggingface."""

        logger.info(
            f"LoRA adapter loading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.load_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter loading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    # 直接从 tensor payload 加载 LoRA adapter。
    # 如果 payload 是 flattened_bucket，会先根据 metadata 还原各个 LoRA tensor，再交给 LoRA manager 注册。
    def load_lora_adapter_from_tensors(
        self, lora_ref: LoRARef, tensors, config_dict, added_tokens_config=None
    ):
        logger.info(f"LoRA adapter loading from tensors starts: {lora_ref}.")
        result = self.lora_manager.load_lora_adapter_from_tensors(
            lora_ref, tensors, config_dict, added_tokens_config
        )
        logger.info(f"LoRA adapter loading from tensors completes: {lora_ref}.")
        return result

    # 卸载一个 LoRA adapter。
    # 这会从 LoRA manager 中移除对应 adapter 的权重和索引，后续请求不能再引用它。
    def unload_lora_adapter(self, lora_ref: LoRARef):
        """Unload a lora adapter that was previously loaded during initialization or dynamic loading."""

        logger.info(
            f"LoRA adapter unloading starts: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        result = self.lora_manager.unload_lora_adapter(lora_ref)

        logger.info(
            f"LoRA adapter unloading completes: {lora_ref}. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

        return result

    @property
    # 读取 Qwen3-Next 这类 hybrid/linear attention 模型的专用配置。
    # 这些属性会影响 KV cache 组织、attention backend 和层级执行方式。
    def qwen3_next_config(self):
        config = self.model_config.hf_config
        if isinstance(config, Qwen3NextConfig):
            return config
        return None

    @property
    # 读取 hybrid lightning 模型配置。
    # 该配置描述哪些层走特殊 linear/hybrid attention 路径。
    def hybrid_lightning_config(self):
        config = self.model_config.hf_config
        if isinstance(config, BailingHybridConfig):
            return config
        return None

    @property
    # 读取 hybrid GDN 模型配置。
    # ModelRunner 通过该属性判断是否需要启用对应 linear attention 运行时支持。
    def hybrid_gdn_config(self):
        config = self.model_config.hf_config.get_text_config()
        if isinstance(
            config,
            Qwen3NextConfig
            | Qwen3_5Config
            | Qwen3_5MoeConfig
            | InternS2PreviewConfig
            | JetNemotronConfig
            | JetVLMConfig,
        ):
            return config
        return None

    @property
    # 读取 Mamba2 配置。
    # Mamba/SSM 类模型和标准 transformer attention 不同，因此需要单独暴露配置给执行层。
    def mamba2_config(self):
        config = self.model_config.hf_config
        if isinstance(config, NemotronHConfig) and self.is_draft_worker:
            # NemotronH MTP draft models have no Mamba layers (pattern like "*E")
            # so they shouldn't use HybridLinearAttnBackend
            pattern = getattr(config, "mtp_hybrid_override_pattern", None)
            if pattern is not None and "M" not in pattern:
                return None
        if isinstance(
            config,
            FalconH1Config
            | NemotronHConfig
            | Lfm2Config
            | Lfm2MoeConfig
            | Lfm2VlConfig,
        ):
            return config
        if isinstance(config, NemotronH_Nano_VL_V2_Config):
            return config.llm_config

        if isinstance(config, GraniteMoeHybridConfig):
            has_mamba = any(
                layer_type == "mamba"
                for layer_type in getattr(config, "layer_types", [])
            )
            if not has_mamba:
                return None
            else:
                return config

        return None

    @property
    # 返回 token pool 的最大容量。
    # Scheduler 和内存池逻辑会用这个值判断还能容纳多少 token/KV cache 条目。
    def max_token_pool_size(self):
        """Return the max token pool size considering hybrid swa settings."""
        if self.is_hybrid_swa:
            return self.full_max_total_num_tokens
        else:
            return self.max_total_num_tokens

    @property
    # 读取 Kimi linear attention 相关配置。
    # 该配置会影响 linear attention registry 和 backend 的选择。
    def kimi_linear_config(self):
        config = self.model_config.hf_config
        if isinstance(config, KimiLinearConfig):
            return config
        return None

    # 从 linear attention registry 中查找当前模型匹配的实现。
    # 如果匹配成功，后续属性会复用该结果获取模型 spec 和配置。
    def _get_linear_attn_registry_result(self):
        if self._linear_attn_registry_cache is _UNSET:
            self._linear_attn_registry_cache = get_linear_attn_config(
                self.model_config.hf_config
            )
        return self._linear_attn_registry_cache

    @property
    # 返回当前 linear attention 模型的 spec。
    # spec 描述模型需要的线性 attention 执行方式和缓存结构。
    def linear_attn_model_spec(self):
        result = self._get_linear_attn_registry_result()
        return result[0] if result else None

    @property
    # 读取 Mamba-like 模型配置。
    # 这类模型使用状态空间或混合模块，执行路径与纯 attention 模型不同。
    def mambaish_config(self):
        existing = (
            self.mamba2_config
            or self.hybrid_gdn_config
            or self.kimi_linear_config
            or self.hybrid_lightning_config
        )
        if existing:
            return existing
        result = self._get_linear_attn_registry_result()
        return result[1] if result else None

    # 决定 KV cache 实际使用的数据类型。
    # auto 模式会参考模型量化配置；显式配置可选择 FP8、BF16、FP4 等类型。
    # KV cache dtype 直接影响显存占用、attention kernel 兼容性和精度。
    def configure_kv_cache_dtype(self):
        if self.server_args.kv_cache_dtype == "auto":
            quant_config = getattr(self.model, "quant_config", None)
            kv_cache_quant_algo = getattr(quant_config, "kv_cache_quant_algo", None)
            if (
                isinstance(kv_cache_quant_algo, str)
                and kv_cache_quant_algo.upper() == "FP8"
            ):
                if _is_hip:
                    self.kv_cache_dtype = fp8_dtype
                    self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                        self.kv_cache_dtype
                    ]
                else:
                    self.kv_cache_dtype = torch.float8_e4m3fn
                    self.server_args.kv_cache_dtype = TORCH_DTYPE_TO_KV_CACHE_STR[
                        self.kv_cache_dtype
                    ]
            else:
                self.kv_cache_dtype = self.dtype
        elif self.server_args.kv_cache_dtype == "fp8_e5m2":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e5m2
        elif self.server_args.kv_cache_dtype == "fp8_e4m3":
            if _is_hip:  # Using natively supported format
                self.kv_cache_dtype = fp8_dtype
            else:
                self.kv_cache_dtype = torch.float8_e4m3fn
        elif self.server_args.kv_cache_dtype in ("bf16", "bfloat16"):
            self.kv_cache_dtype = torch.bfloat16
        elif self.server_args.kv_cache_dtype == "fp4_e2m1":
            if hasattr(torch, "float4_e2m1fn_x2"):
                self.kv_cache_dtype = torch.float4_e2m1fn_x2
                logger.warning(f"FP4 (E2M1) KV Cache might lead to a accuracy drop!")
            else:
                logger.warning(
                    f"--kv-cache-dtype falls back to 'auto' because this torch version does not support torch.float4_e2m1fn_x2"
                )
                self.kv_cache_dtype = self.dtype
        else:
            raise ValueError(
                f"Unsupported kv_cache_dtype: {self.server_args.kv_cache_dtype}."
            )

    # 通过一个很小的矩阵乘法触发 cuBLAS 初始化。
    # 这样真实请求第一次执行 matmul 时不会额外承担库初始化延迟。
    def init_cublas(self):
        """We need to run a small matmul to init cublas. Otherwise, it will raise some errors later."""
        dtype = torch.float16
        device = "cuda"
        a = torch.ones((16, 16), dtype=dtype, device=device)
        b = torch.ones((16, 16), dtype=dtype, device=device)
        c = a @ b
        return c

    # 创建当前 ModelRunner 使用的 attention backend。
    # 普通路径只创建一个 backend；PDMux 会创建 decode backend 组；two-batch-overlap 会包一层 TBO backend。
    # attention backend 负责为每次 forward 准备 page table、seq lens、workspace 等 kernel metadata。
    def init_attention_backend(self):
        """Init attention kernel backend."""
        # PDMux 会为 decode 准备一组 attention backend。
        # 不同 SM group 可以使用不同 decode backend，从而支持更细粒度的并行调度。
        if self.server_args.enable_pdmux:
            self.attn_backend = self._get_attention_backend(init_new_workspace=True)
            self.decode_attn_backend_group = []
            for _ in range(self.server_args.sm_group_num):
                self.decode_attn_backend_group.append(self._get_attention_backend())
            self.decode_attn_backend = self.decode_attn_backend_group[0]
        # two-batch-overlap 会把 attention backend 包装成 TBO backend。
        # 目的是让两个 batch 的部分计算阶段重叠，提高 GPU 利用率。
        elif self.server_args.enable_two_batch_overlap and not self.is_draft_worker:
            self.attn_backend = TboAttnBackend.init_new(self._get_attention_backend)
        else:
            self.attn_backend = self._get_attention_backend()

    # 解析 prefill 与 decode 的 attention backend 配置。
    # 如果两者不同，会创建 HybridAttnBackend，把 prefill 和 decode 分发到不同实现。
    # 该函数还会把最终选择写回 global server args，便于其他模块读取。
    def _get_attention_backend(self, init_new_workspace: bool = False):
        """Init attention kernel backend."""
        draft_attn_backend = self.server_args.speculative_draft_attention_backend
        if self.is_draft_worker and draft_attn_backend:
            logger.warning(
                f"Overriding draft attention backend to {draft_attn_backend}."
            )
            return self._get_attention_backend_from_str(
                draft_attn_backend,
                init_new_workspace=init_new_workspace,
            )

        (
            self.prefill_attention_backend_str,
            self.decode_attention_backend_str,
        ) = self.server_args.get_attention_backends()

        # prefill 和 decode 的最优 attention backend 可能不同。
        # 当两者不同，HybridAttnBackend 会在运行时根据 forward mode 选择对应实现。
        if self.decode_attention_backend_str != self.prefill_attention_backend_str:
            from sglang.srt.layers.attention.hybrid_attn_backend import (
                HybridAttnBackend,
            )

            attn_backend = HybridAttnBackend(
                self,
                decode_backend=self._get_attention_backend_from_str(
                    self.decode_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
                prefill_backend=self._get_attention_backend_from_str(
                    self.prefill_attention_backend_str,
                    init_new_workspace=init_new_workspace,
                ),
            )
            logger.info(
                f"Using hybrid attention backend for decode and prefill: "
                f"decode_backend={self.decode_attention_backend_str}, "
                f"prefill_backend={self.prefill_attention_backend_str}."
            )
            logger.warning(
                "Warning: Attention backend specified by --attention-backend or default backend might be overridden."
                "The feature of hybrid attention backend is experimental and unstable. Please raise an issue if you encounter any problem."
            )
        else:
            attn_backend = self._get_attention_backend_from_str(
                self.server_args.attention_backend,
                init_new_workspace=init_new_workspace,
            )

        (
            get_global_server_args().prefill_attention_backend,
            get_global_server_args().decode_attention_backend,
        ) = (self.prefill_attention_backend_str, self.decode_attention_backend_str)
        return attn_backend

    # 按字符串名称从 ATTENTION_BACKENDS 注册表中实例化具体 backend。
    # 创建后会经过 attn_backend_wrapper 包装，使外层调用拥有统一接口。
    def _get_attention_backend_from_str(
        self, backend_str: str, init_new_workspace: bool = False
    ):
        if backend_str not in ATTENTION_BACKENDS:
            raise ValueError(f"Invalid attention backend: {backend_str}")
        self.init_new_workspace = init_new_workspace
        full_attention_backend = ATTENTION_BACKENDS[backend_str](self)
        return attn_backend_wrapper(self, full_attention_backend)

    # 在 graph capture 和真实请求前预热关键 kernel。
    # FlashInfer autotune、PP parallel DeepGEMM warmup 等逻辑都在这里触发。
    # 这一步的目标是把编译、调优和 workspace 初始化尽量移到服务启动阶段。
    def kernel_warmup(self):
        """Warmup and tune kernels before cuda graph capture."""
        if self.device != "cuda":
            return

        if self._should_run_flashinfer_autotune():
            self._flashinfer_autotune()

        if (
            envs.SGLANG_PP_PARALLEL_DEEPGEMM_WARMUP.get()
            and deep_gemm_wrapper.ENABLE_JIT_DEEPGEMM
            and self.pp_size > 1
            and not self.spec_algorithm.is_speculative()
        ):
            from sglang.srt.layers.deep_gemm_wrapper.compile_utils import (
                pp_parallel_deep_gemm_warmup,
            )

            pp_parallel_deep_gemm_warmup(self)

    # 预初始化 FlashInfer allreduce fusion workspace。
    # 这一步必须发生在 CUDA graph capture 前，避免 graph 捕获过程中触发 broadcast/barrier 等 collective 导致死锁。
    def _pre_initialize_flashinfer_allreduce_workspace(self):
        """Pre-initialize flashinfer allreduce fusion workspaces.

        Must run before CUDA graph capture to avoid collective operations
        (broadcasts, barriers) inside the graph capture context, which can
        deadlock with custom_all_reduce.register_graph_buffers.
        """
        if not self.server_args.enable_flashinfer_allreduce_fusion:
            return

        from sglang.srt.layers.communicator import FUSE_ALLREDUCE_MAX_BATCH_SIZE
        from sglang.srt.layers.flashinfer_comm_fusion import (
            pre_initialize_workspaces,
        )

        pre_initialize_workspaces(
            max_token_num=FUSE_ALLREDUCE_MAX_BATCH_SIZE,
            hidden_dim=self.model_config.hidden_size,
            dtype=self.dtype,
        )

    # 判断当前配置是否应该运行 FlashInfer autotune。
    # 函数会排除禁用开关、不支持的 MoE backend、低于 sm90 的 GPU，以及 draft worker 等不适合调优的场景。
    def _should_run_flashinfer_autotune(self) -> bool:
        """Check if flashinfer autotune should be run."""
        if self.server_args.disable_flashinfer_autotune:
            return False

        # CuteDSL v1 (cutedsl runner + deepep a2a) bypasses MoeRunner and must not
        # be autotuned -- its _dummy_run would dispatch more tokens per rank than
        # SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK, tripping a DeepEP assert.
        # Read server_args directly to avoid depending on initialize_moe_config()
        # having already populated the MoE backend globals.
        if (
            self.server_args.moe_runner_backend == "flashinfer_cutedsl"
            and self.server_args.moe_a2a_backend == "deepep"
        ):
            return False

        backend_str = self.server_args.moe_runner_backend

        # TODO smor- support other cases for flashinfer autotune, such as, mamba backend

        if backend_str not in [
            "flashinfer_trtllm",
            # TODO: Enable for flashinfer_trtllm_routed once https://github.com/flashinfer-ai/flashinfer/issues/2749 is fixed.
            # "flashinfer_trtllm_routed",
            "flashinfer_mxfp4",
            "flashinfer_cutedsl",
            # TODO: flashinfer_cutlass will cause some flashinfer compilation errors. To be fixed.
            # "flashinfer_cutlass",
        ]:
            return False

        major, _ = torch.cuda.get_device_capability()
        if major < 9:
            return False

        if self.spec_algorithm.is_speculative():
            return not self.is_draft_worker

        return True

    # 执行 FlashInfer autotune。
    # 它会选择或创建 cache 文件，在 forward_stream 上跑 _dummy_run，让 FlashInfer 为当前模型、并行配置和硬件生成更合适的 kernel 参数。
    def _flashinfer_autotune(self):
        """Run flashinfer autotune."""
        from flashinfer.autotuner import autotune

        cache_path = self._flashinfer_autotune_cache_path()
        if envs.SGLANG_FLASHINFER_AUTOTUNE_CACHE.get():
            autotune_cache = cache_path
            logger.info("Running FlashInfer autotune with cache: %s", autotune_cache)
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            runs_dir = cache_path.parent / "runs"
            runs_dir.mkdir(parents=True, exist_ok=True)
            autotune_cache = (
                runs_dir / f"{cache_path.stem}.{timestamp}{cache_path.suffix}"
            )
            logger.info(
                "Running FlashInfer autotune (cache reuse DISABLED via "
                "SGLANG_FLASHINFER_AUTOTUNE_CACHE=0); writing fresh result to: %s",
                autotune_cache,
            )

        # Run warmup on the non-default stream to avoid NCCL 2.29+ cudaMemcpyBatchAsync
        # calls on default stream (unsupported by CUDA) when --enable-symm-mem is used.
        self.forward_stream.wait_stream(torch.cuda.current_stream())
        with torch.get_device_module(self.device).stream(self.forward_stream):
            with torch.inference_mode(), autotune(True, cache=str(autotune_cache)):
                self._dummy_run(batch_size=self.req_to_token_pool.size)
        torch.cuda.current_stream().wait_stream(self.forward_stream)
        logger.info("FlashInfer autotune completed.")

    # 生成 FlashInfer autotune 结果的缓存路径。
    # 路径由模型、dtype、量化、MoE backend、TP/PP/DP/EP 和 GPU 架构共同决定，避免不同运行配置复用错误调优结果。
    def _flashinfer_autotune_cache_path(self) -> Path:
        import flashinfer

        major, minor = torch.cuda.get_device_capability(self.device)
        arch = f"sm{major}{minor}"
        flashinfer_version = getattr(flashinfer, "__version__", "unknown")

        server_args = self.server_args
        model_key = "|".join(
            [
                str(server_args.model_path),
                str(self.dtype),
                str(server_args.quantization),
                str(server_args.moe_runner_backend),
                str(self.tp_size),
                str(self.pp_size),
                str(self.dp_size),
                str(self.moe_ep_size),
                str(self.model_config.hf_config.__class__.__name__),
            ]
        )
        cache_key = hashlib.sha256(model_key.encode()).hexdigest()[:16]
        cache_dir = (
            Path(envs.SGLANG_CACHE_DIR.get())
            / "flashinfer"
            / "autotune"
            / flashinfer_version
            / arch
            / cache_key
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return (
            cache_dir
            / f"rank_tp{self.tp_rank}_pp{self.pp_rank}_dp{self.dp_rank or 0}.json"
        )

    # 构造一个假的 ForwardBatch 并执行一次前向。
    # 它用于 FlashInfer autotune、graph capture 前 warmup、DeepGEMM warmup 等场景。
    # 重点是它会模拟 decode/extend/speculative 等不同模式需要的张量形状和 metadata。
    def _dummy_run(
        self,
        batch_size: int,
        run_ctx=None,
        forward_mode_override: Optional[ForwardMode] = None,
    ):
        """Run a dummy forward pass for warmup/profiling.

        ``forward_mode_override`` forces EXTEND/DECODE regardless of
        ``is_generation`` (used by the PP-parallel DeepGEMM warmup).
        """
        if forward_mode_override is not None:
            capture_forward_mode = forward_mode_override
        elif self.is_generation:
            capture_forward_mode = ForwardMode.DECODE
        else:
            capture_forward_mode = ForwardMode.EXTEND
        capture_hidden_mode = CaptureHiddenMode.NULL
        num_tokens_per_bs = 1
        if self.spec_algorithm.is_speculative():
            if self.is_draft_worker:
                if not self.spec_algorithm.supports_target_verify_for_draft():
                    raise RuntimeError("This should not happen")
            capture_forward_mode = ForwardMode.TARGET_VERIFY
            num_tokens_per_bs = (
                self.spec_algorithm.get_num_tokens_per_bs_for_target_verify(
                    self.server_args.speculative_num_draft_tokens, self.is_draft_worker
                )
            )

        if self.server_args.enable_return_hidden_states:
            capture_hidden_mode = CaptureHiddenMode.FULL

        num_tokens = batch_size * num_tokens_per_bs

        # Keep warmup aligned with scheduler MLP-sync padding.
        if require_mlp_sync(self.server_args):
            attn_tp_size = get_attention_tp_size()
            if attn_tp_size > 1 and num_tokens % attn_tp_size != 0:
                num_tokens = ceil_align(num_tokens, attn_tp_size)
                batch_size = num_tokens // num_tokens_per_bs

        seq_len_fill_value = self.attn_backend.get_cuda_graph_seq_len_fill_value()

        if self.server_args.enable_torch_compile:
            set_torch_compile_config()
            should_disable_torch_compile = not getattr(
                self.model, "_can_torch_compile", True
            )
            if should_disable_torch_compile:
                log_info_on_rank0(
                    logger,
                    "Transformers backend model reports it is not torch.compile "
                    "compatible (e.g. dynamic rope scaling). Disabling torch.compile.",
                )
                self.server_args.enable_torch_compile = False

        # NOTE: aux hidden state capture (eagle3/dflash) is already
        # configured by init_aux_hidden_state_capture() in initialize().

        require_mlp_tp_gather_ = require_mlp_tp_gather(self.server_args)
        if require_gathered_buffer(self.server_args):
            assert require_mlp_tp_gather_ or require_attn_tp_gather(self.server_args)

        buffers: DecodeInputBuffers = DecodeInputBuffers.create(
            device=self.device,
            max_bs=batch_size,
            max_num_token=num_tokens,
            hidden_size=self.model_config.hidden_size,
            vocab_size=self.model_config.vocab_size,
            dtype=self.model_config.dtype,
            dp_size=self.server_args.dp_size,
            pp_size=self.server_args.pp_size,
            is_encoder_decoder=self.model_config.is_encoder_decoder,
            require_mlp_tp_gather=require_mlp_tp_gather_,
            seq_len_fill_value=seq_len_fill_value,
            encoder_len_fill_value=(
                getattr(self.model_config.hf_config, "max_source_positions", 0)
                if self.model_config.is_encoder_decoder
                else 0
            ),
            num_tokens_per_bs=num_tokens_per_bs,
            cache_loc_dtype=torch.int64,
            enable_mamba_track=False,
            hc_hidden_size=getattr(self.model_config, "hc_hidden_size", None),
        )
        buffers.num_token_non_padded[...] = num_tokens

        # For extend mode
        if capture_forward_mode == ForwardMode.EXTEND:
            extend_prefix_lens_cpu = [0] * batch_size
            extend_seq_lens_cpu = [seq_len_fill_value] * batch_size
            extend_num_tokens = num_tokens
            extend_seq_lens = torch.full(
                (batch_size,), seq_len_fill_value, dtype=torch.int32, device=self.device
            )
            extend_prefix_lens = torch.zeros(
                (batch_size,), dtype=torch.int32, device=self.device
            )
            extend_start_loc = torch.arange(
                0, num_tokens, num_tokens_per_bs, dtype=torch.int32, device=self.device
            )
        else:
            extend_prefix_lens_cpu = None
            extend_seq_lens_cpu = None
            extend_num_tokens = None
            extend_seq_lens = None
            extend_prefix_lens = None
            extend_start_loc = None

        if self.server_args.pp_size > 1:
            # PP0 already cp-split hidden_states before send.
            pp_hidden_tokens = num_tokens
            if (
                capture_forward_mode == ForwardMode.EXTEND
                and self.pp_rank != 0
                and self.attn_cp_size > 1
            ):
                pp_hidden_tokens = num_tokens // self.attn_cp_size
            pp_proxy_tensors = PPProxyTensors(
                {k: v[:pp_hidden_tokens] for k, v in buffers.pp_proxy_tensors.items()}
            )

        if require_mlp_tp_gather_:
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.server_args.dp_size,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens] * self.server_args.dp_size,
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            global_dp_buffer_len = num_tokens * self.server_args.dp_size
            global_num_tokens_cpu = [num_tokens] * self.server_args.dp_size
        elif require_attn_tp_gather(self.server_args):
            buffers.global_num_tokens_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            buffers.global_num_tokens_for_logprob_gpu.copy_(
                torch.tensor(
                    [num_tokens],
                    dtype=torch.int32,
                    device=self.device,
                )
            )
            global_dp_buffer_len = num_tokens
            global_num_tokens_cpu = [num_tokens]
        else:
            global_dp_buffer_len = None
            global_num_tokens_cpu = None

        # 为 dummy run 构造 speculative decoding 的验证信息。
        # EAGLE、DFLASH、ngram 等算法需要不同的 spec_info 形状；warmup 时不关心真实 token，只需要 metadata 能驱动 kernel 初始化。
        def get_spec_info():
            spec_info = None
            if self.spec_algorithm.is_eagle() or self.spec_algorithm.is_standalone():
                from sglang.srt.speculative.eagle_info import EagleVerifyInput

                if self.is_draft_worker:
                    raise RuntimeError("This should not happen.")
                else:
                    spec_info = EagleVerifyInput(
                        draft_token=None,
                        custom_mask=buffers.custom_mask,
                        positions=None,
                        retrieve_index=None,
                        retrieve_next_token=None,
                        retrieve_next_sibling=None,
                        retrieve_cum_len=None,
                        spec_steps=self.server_args.speculative_num_steps,
                        topk=self.server_args.speculative_eagle_topk,
                        draft_token_num=self.server_args.speculative_num_draft_tokens,
                        capture_hidden_mode=CaptureHiddenMode.FULL,
                        seq_lens_sum=None,
                        seq_lens_cpu=None,
                    )
            elif self.spec_algorithm.is_dflash():
                from sglang.srt.speculative.dflash_info import DFlashVerifyInput

                # Dummy warmup only needs shape metadata; avoid forcing custom-mask mode.
                spec_info = DFlashVerifyInput(
                    draft_token=None,
                    positions=None,
                    draft_token_num=self.server_args.speculative_num_draft_tokens,
                    custom_mask=None,
                    capture_hidden_mode=(
                        CaptureHiddenMode.NULL
                        if self.is_draft_worker
                        else CaptureHiddenMode.FULL
                    ),
                )

            elif self.spec_algorithm.is_ngram():
                from sglang.srt.speculative.ngram_info import NgramVerifyInput

                spec_info = NgramVerifyInput(
                    draft_token=None,
                    tree_mask=buffers.custom_mask,
                    positions=None,
                    retrieve_index=None,
                    retrieve_next_token=None,
                    retrieve_next_sibling=None,
                    draft_token_num=num_tokens_per_bs,
                )
                spec_info.capture_hidden_mode = CaptureHiddenMode.NULL

            return spec_info

        spec_info = get_spec_info()
        if capture_hidden_mode != CaptureHiddenMode.FULL:
            capture_hidden_mode = (
                spec_info.capture_hidden_mode if spec_info else CaptureHiddenMode.NULL
            )

        if self.server_args.enable_lora:
            lora_ids = [None] * batch_size
        else:
            lora_ids = None

        forward_batch = ForwardBatch(
            forward_mode=capture_forward_mode,
            batch_size=batch_size,
            input_ids=buffers.input_ids,
            req_pool_indices=buffers.req_pool_indices,
            seq_lens=buffers.seq_lens,
            seq_lens_cpu=buffers.seq_lens_cpu,
            next_token_logits_buffer=buffers.next_token_logits_buffer,
            orig_seq_lens=buffers.seq_lens,
            out_cache_loc=buffers.out_cache_loc,
            seq_lens_sum=buffers.seq_lens.sum().item(),
            encoder_lens=buffers.encoder_lens,
            return_logprob=False,
            positions=buffers.positions,
            extend_num_tokens=extend_num_tokens,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_start_loc=extend_start_loc,
            extend_prefix_lens_cpu=extend_prefix_lens_cpu,
            extend_seq_lens_cpu=extend_seq_lens_cpu,
            global_num_tokens_gpu=buffers.global_num_tokens_gpu,
            global_num_tokens_cpu=global_num_tokens_cpu,
            global_num_tokens_for_logprob_gpu=buffers.global_num_tokens_for_logprob_gpu,
            dp_padding_mode=DpPaddingMode.get_default_mode_in_cuda_graph(),
            global_dp_buffer_len=global_dp_buffer_len,
            mrope_positions=buffers.mrope_positions,
            spec_algorithm=self.spec_algorithm,
            spec_info=spec_info,
            capture_hidden_mode=capture_hidden_mode,
            num_token_non_padded=buffers.num_token_non_padded,
            global_forward_mode=capture_forward_mode,
            lora_ids=lora_ids,
        )

        if lora_ids is not None:
            self.lora_manager.prepare_lora_batch(forward_batch)

        # 为当前 batch 初始化 attention metadata。
        # 这通常包括 seq_lens、prefix_lens、page table、cache loc、custom mask、workspace 指针等 kernel 输入。
        self.attn_backend.init_forward_metadata(forward_batch)

        # dummy run 的单次前向函数。
        # 它重置 DP padding 状态、准备 PP proxy 或 embedding kwargs，然后调用 model.forward 触发 kernel 编译、autotune 或 graph capture 前的预热。
        def run_once():
            forward_batch.dp_local_start_pos = forward_batch.dp_local_num_tokens = None
            set_dp_buffer_len(
                global_dp_buffer_len,
                num_tokens,
                forward_batch.dp_padding_mode.is_max_len(),
            )
            set_is_extend_in_batch(False)

            kwargs = {}
            if (
                self.server_args.pp_size > 1
                and "pp_proxy_tensors"
                in inspect.signature(self.model.forward).parameters
            ):
                kwargs["pp_proxy_tensors"] = PPProxyTensors(
                    {k: v.clone() for k, v in pp_proxy_tensors.tensors.items()}
                )
            if not self.is_generation:
                kwargs["get_embedding"] = True

            logits_output_or_pp_proxy_tensors = self.model.forward(
                buffers.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
            return logits_output_or_pp_proxy_tensors

        torch.get_device_module(self.device).synchronize()
        self.tp_group.barrier()
        # 建立 ForwardContext。
        # 模型层内部不直接持有 ModelRunner，而是通过上下文读取当前 attention backend 和相关运行状态。
        with forward_context(ForwardContext(attn_backend=self.attn_backend)):
            with torch.inference_mode(), run_ctx or empty_context():
                run_once()

    # 如果模型启用了 ngram embedding，这里初始化 token table 与模块内部 buffer。
    # ngram embedding 依赖请求维度的 token 历史，因此需要和 req_to_token_pool 的容量对齐。
    def maybe_init_ngram_embedding(self):
        self.use_ngram_embedding = self.model_config.use_ngram_embedding
        if self.use_ngram_embedding:
            from sglang.srt.layers.n_gram_embedding import NgramEmbedding

            # Sized to mirror req_to_token (indexed by req_pool_idx).
            self.token_table = torch.empty(
                self.req_to_token_pool.req_to_token.shape[0],
                self.model_config.context_len,
                dtype=torch.int32,
                device=self.device,
            )
            chunked_prefill_size = self.server_args.chunked_prefill_size
            assert (
                chunked_prefill_size is not None and chunked_prefill_size > 0
            ), "Ngram embedding requires chunked prefill to be enabled (chunked_prefill_size > 0)"
            for module in self.model.modules():
                if isinstance(module, NgramEmbedding):
                    module.init_buffers(
                        self.max_running_requests, chunked_prefill_size, self.device
                    )

    # 每次采样出新 token 后，把 token 写回 ngram token table。
    # 这样下一轮 forward 的 ngram embedding 能看到最新生成结果。
    def maybe_update_ngram_token_table(
        self,
        next_token_ids: torch.Tensor,
        forward_batch: "ForwardBatch",
    ):
        """Update the ngram embedding token table after sampling."""
        ngram_embedding_info = forward_batch.ngram_embedding_info
        if ngram_embedding_info is None:
            return
        ngram_embedding_info.out_column_starts[: forward_batch.batch_size] = (
            forward_batch.seq_lens
        )
        ngram_embedding_info.out_req_lens[: forward_batch.batch_size] = 1
        update_token_table(
            ne_token_table=ngram_embedding_info.token_table,
            tokens=next_token_ids.to(torch.int32),
            row_indices=forward_batch.req_pool_indices,
            column_starts=ngram_embedding_info.out_column_starts,
            req_lens=torch.ones_like(ngram_embedding_info.out_column_starts),
            ignore_tokens=None,
        )

    # 捕获设备 graph，主要用于 decode 阶段的固定形状加速。
    # 函数会根据设备类型选择 CUDA/CPU/NPU/out-of-tree graph runner。
    # 捕获成功后，_forward_raw 可以在形状匹配时直接 replay graph，减少 Python 和 kernel launch 开销。
    def init_device_graphs(self):
        """Capture device graphs."""
        self.graph_runner = None
        self.graph_mem_usage = 0

        if not self.is_generation:
            # TODO: Currently, cuda graph only captures decode steps, which only exists for generation models
            return

        if self.server_args.model_impl.lower() == ModelImpl.MINDSPORE:
            return

        if self.device != "cpu" and self.server_args.disable_cuda_graph:
            return

        if self.device == "cpu" and not self.server_args.enable_torch_compile:
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        graph_backend = defaultdict(
            lambda: f"{current_platform.device_name} graph",
            {
                "cuda": "cuda graph",
                "musa": "cuda graph",
                "cpu": "cpu graph",
                "npu": "npu graph",
            },
        )
        logger.info(
            f"Capture {graph_backend[self.device]} begin. This can take up to several minutes. avail mem={before_mem:.2f} GB"
        )
        if current_platform.is_out_of_tree():
            GraphRunnerCls = current_platform.get_graph_runner_cls()
            self.graph_runner = GraphRunnerCls(self)
        else:
            graph_runners = defaultdict(
                lambda: CudaGraphRunner,
                {
                    "cpu": CPUGraphRunner,
                    "npu": NPUGraphRunner,
                },
            )
            self.graph_runner = graph_runners[self.device](self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        self.graph_mem_usage = before_mem - after_mem
        logger.info(
            f"Capture {graph_backend[self.device]} end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={self.graph_mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    # 初始化更细粒度的 piecewise CUDA graph。
    # 它会遍历模型 layers，收集 attention 层、MoE 层和相关 fusion/indexer，用于局部图捕获。
    # 这种方式比整图 capture 更灵活，适合一些动态性更强的 prefill/extend 路径。
    def init_piecewise_cuda_graphs(self, force_for_draft_worker: bool = False):
        """Initialize piecewise CUDA graph runner."""
        self.piecewise_cuda_graph_runner = None

        if self.server_args.disable_piecewise_cuda_graph:
            logger.info(
                "Disable piecewise CUDA graph because --disable-piecewise-cuda-graph is set"
            )
            return

        # Draft models skip here during __init__; the eagle worker calls
        # this method explicitly (force_for_draft_worker=True) after
        # init_lm_head so graphs capture the final embedding weights.
        if self.is_draft_worker and not force_for_draft_worker:
            return

        # Disable piecewise CUDA graph for non-language models
        if not hasattr(self.model, "model"):
            logger.warning(
                "Disable piecewise CUDA graph because the model is not a language model"
            )
            return

        # Disable piecewise CUDA graph for non capture size
        if not self.server_args.piecewise_cuda_graph_tokens:
            logger.warning(
                "Disable piecewise CUDA graph because the capture size is not set"
            )
            return

        # Collect attention layers and moe layers from the model
        self.model.model = resolve_language_model(self.model)
        language_model = getattr(self.model, "language_model", self.model)

        # Resolve model with layers: handle CausalLM wrapper (.model.layers) and direct TextModel (.layers)
        if hasattr(language_model, "model") and hasattr(language_model.model, "layers"):
            layer_model = language_model.model
        elif hasattr(language_model, "layers"):
            layer_model = language_model
        else:
            logger.warning(
                "Disable piecewise CUDA graph because the model does not have a 'layers' attribute"
            )
            return

        self.attention_layers = []
        self.moe_layers = []
        self.moe_fusions = []
        self.dsa_indexers = []
        for layer in layer_model.layers:
            attn_layer = None
            if hasattr(layer, "self_attn"):
                if hasattr(layer.self_attn, "attn"):
                    attn_layer = layer.self_attn.attn
                elif hasattr(layer.self_attn, "attn_mqa"):
                    # For DeepSeek model
                    attn_layer = layer.self_attn.attn_mqa
            # For hybrid model
            elif hasattr(layer, "attn"):
                attn_layer = layer.attn
            elif hasattr(layer, "linear_attn"):
                if hasattr(layer.linear_attn, "attn"):
                    attn_layer = layer.linear_attn.attn
                else:
                    attn_layer = layer.linear_attn
            # For InternVL model
            elif hasattr(layer, "attention"):
                if hasattr(layer.attention, "attn"):
                    attn_layer = layer.attention.attn
            # For NemotronH and similar hybrid models using 'mixer' attribute
            elif hasattr(layer, "mixer"):
                if hasattr(layer.mixer, "attn"):
                    attn_layer = layer.mixer.attn
                elif hasattr(layer, "_forward_mamba"):
                    # Mamba layer with split op support - store the layer itself
                    attn_layer = layer

            if attn_layer is not None:
                self.attention_layers.append(attn_layer)
            elif hasattr(layer, "mixer"):
                self.attention_layers.append(None)

            moe_block = None
            moe_fusion = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "experts"):
                moe_block = layer.mlp.experts
                moe_fusion = layer.mlp
            if hasattr(layer, "block_sparse_moe") and hasattr(
                layer.block_sparse_moe, "experts"
            ):
                moe_block = layer.block_sparse_moe.experts
                moe_fusion = layer.block_sparse_moe
            if hasattr(layer, "moe") and hasattr(layer.moe, "experts"):
                moe_block = layer.moe.experts
                moe_fusion = layer.moe
            # For NemotronH MoE layers using 'mixer' attribute
            if hasattr(layer, "mixer") and hasattr(layer.mixer, "experts"):
                moe_block = layer.mixer.experts
                moe_fusion = layer.mixer
            self.moe_layers.append(moe_block)
            self.moe_fusions.append(moe_fusion)
            # NSA indexers (None for layers without NSA)
            dsa_indexer = None
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "indexer"):
                dsa_indexer = layer.self_attn.indexer
            self.dsa_indexers.append(dsa_indexer)

        if len(self.attention_layers) < self.model_config.num_hidden_layers:
            # TODO(yuwei): support Non-Standard GQA
            log_info_on_rank0(
                logger,
                "Disable piecewise CUDA graph because some layers do not apply Standard GQA",
            )
            return

        tic = time.perf_counter()
        before_mem = get_available_gpu_memory(self.device, self.gpu_id)
        logger.info(
            f"Capture piecewise CUDA graph begin. avail mem={before_mem:.2f} GB"
        )

        if self.server_args.enable_breakable_cuda_graph:
            # Experimental feature
            self.piecewise_cuda_graph_runner = BreakableCudaGraphRunner(self)
        else:
            self.piecewise_cuda_graph_runner = PiecewiseCudaGraphRunner(self)

        after_mem = get_available_gpu_memory(self.device, self.gpu_id)
        mem_usage = before_mem - after_mem
        logger.info(
            f"Capture piecewise CUDA graph end. Time elapsed: {time.perf_counter() - tic:.2f} s. "
            f"mem usage={mem_usage:.2f} GB. avail mem={after_mem:.2f} GB."
        )

    # 初始化 CPU 线程绑核策略。
    # 在 CPU 或混合后端下，线程绑定会影响推理吞吐和延迟稳定性。
    def init_threads_binding(self):
        omp_cpuids = os.environ.get("SGLANG_CPU_OMP_THREADS_BIND", "all")
        cpu_ids_by_node = get_cpu_ids_by_node()
        n_numa_node = len(cpu_ids_by_node)
        if omp_cpuids == "all":
            assert self.tp_size <= n_numa_node, (
                f"SGLANG_CPU_OMP_THREADS_BIND is not set, in this case, "
                f"tp_size {self.tp_size} should be smaller than or equal to number of numa node on the machine {n_numa_node}. "
                f"If you need tp_size to be larger than number of numa node, please set the CPU cores for each tp rank via SGLANG_CPU_OMP_THREADS_BIND explicitly. "
                f"For example, on a machine with 2 numa nodes, where core 0-31 are on numa node 0 and core 32-63 are on numa node 1, "
                f"it is suggested to use -tp 2 and bind tp rank 0 to core 0-31 and tp rank 1 to core 32-63. "
                f"This is the default behavior if SGLANG_CPU_OMP_THREADS_BIND is not set and it is the same as setting SGLANG_CPU_OMP_THREADS_BIND=0-31|32-63. "
                f"If you do need tp_size to be larger than the number of numa nodes, you could set SGLANG_CPU_OMP_THREADS_BIND explicitly for example SGLANG_CPU_OMP_THREADS_BIND=0-15|16-31|32-47|48-63 and run with -tp 4. "
                f"If you don't want each tp rank to use all the cores on one numa node, you could set for example SGLANG_CPU_OMP_THREADS_BIND=0-15|32-47 and run with -tp 2."
            )
            if self.tp_size < n_numa_node:
                logger.warning(
                    f"Detected the current machine has {n_numa_node} numa nodes available, but tp_size is set to {self.tp_size}, so only {self.tp_size} numa nodes are used."
                )
            self.local_omp_cpuid = cpu_ids_by_node[self.tp_rank]
        else:
            threads_bind_list = omp_cpuids.split("|")
            assert self.tp_size == len(threads_bind_list), (
                f"SGLANG_CPU_OMP_THREADS_BIND setting must be aligned with TP size parameter ({self.tp_size}). "
                f"Please double check your settings."
            )
            self.local_omp_cpuid = threads_bind_list[self.tp_rank]
            if self.tp_size > n_numa_node:
                logger.warning(
                    f"TP size ({self.tp_size})is larger than numa node number ({n_numa_node}), "
                    f"in this case the available memory amount of each rank cannot be determined in prior. "
                    f"Please set proper `--max-total-tokens` to avoid the out-of-memory error."
                )

    # 对支持 torch tensor parallel 的模型应用 TP 切分策略。
    # 这通常发生在模型加载完成之后、正式服务之前。
    def apply_torch_tp(self):
        logger.info(f"Enabling torch tensor parallelism on {self.tp_size} devices.")
        from sglang.srt.layers.model_parallel import tensor_parallel

        device_mesh = torch.distributed.init_device_mesh(self.device, (self.tp_size,))
        tensor_parallel(self.model, device_mesh)

    # 切换当前 decode attention backend。
    # PDMux 场景会维护多个 decode backend，运行时根据 stream/group 选择其中一个执行。
    def update_decode_attn_backend(self, stream_idx: int):
        self.decode_attn_backend = self.decode_attn_backend_group[stream_idx]

    # decode 模式通常每个请求只生成一个 token。
    # 函数先准备模型特定 metadata 和 attention metadata，然后调用 model.forward。
    # decode 形状相对稳定，因此是 CUDA graph replay 的主要优化对象。
    def forward_decode(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        # Set extra arguments
        pdmux_override = False
        if not skip_attn_backend_init:
            if hasattr(self.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                self.model.prepare_forward_batch(forward_batch)
            if self.server_args.enable_pdmux:
                # PDMux decode 路径使用专门的 decode attention backend。
                # decode metadata 与 prefill metadata 的形状和 kernel 需求不同，因此单独初始化。
                self.decode_attn_backend.init_forward_metadata(forward_batch)
                # PDmux selects a per-stream backend; publish it to model-layer
                # readers via the active ForwardContext so RadixAttention etc.
                # dispatch against the right backend for this forward.
                pdmux_override = True
            else:
                self.attn_backend.init_forward_metadata(forward_batch)
        # FIXME: add pp_proxy_tensors arg to all models
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors

        # Launch forward
        ctx = (
            self.device_timer.wrap(metadata={"category": "decode"})
            if self.device_timer
            else contextlib.nullcontext()
        )

        # decode 路径中真正调用模型的一小段闭包。
        # 外层已经准备好 attention metadata 和计时上下文；闭包只负责把 input_ids、positions、ForwardBatch 以及 PP kwargs 交给 model.forward。
        def _do_forward():
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

        with ctx:
            if pdmux_override:
                with forward_context(
                    ForwardContext(attn_backend=self.decode_attn_backend)
                ):
                    return _do_forward()
            return _do_forward()

    # extend/prefill 模式处理 prompt token 或新增上下文 token。
    # 这里会准备 PP proxy、input_embeds、image/multimodal replacement embeds、embedding mode 等 kwargs。
    # 如果 piecewise CUDA graph 覆盖当前 batch，会优先 replay；否则走普通 model.forward。
    def forward_extend(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors=None,
    ) -> Tuple[
        Union[LogitsProcessorOutput, PPProxyTensors, EmbeddingPoolerOutput], bool
    ]:
        # Setup extra arguments
        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        if forward_batch.input_embeds is not None:
            kwargs["input_embeds"] = forward_batch.input_embeds.bfloat16()
        if (
            forward_batch.replace_embeds is not None
            and forward_batch.replace_positions is not None
        ):
            # Token embedding overrides: get base embeddings, scatter replacements
            if "input_embeds" not in kwargs:
                embed_layer = self.model.get_input_embeddings()
                kwargs["input_embeds"] = embed_layer(forward_batch.input_ids)
            kwargs["input_embeds"][forward_batch.replace_positions] = (
                forward_batch.replace_embeds.to(kwargs["input_embeds"].dtype)
            )
        if not self.is_generation:
            kwargs["get_embedding"] = True

        # Check piecewies cuda graph
        can_run_graph = (
            self.piecewise_cuda_graph_runner is not None
            and self.piecewise_cuda_graph_runner.can_run(forward_batch)
        )
        if can_run_graph:
            # TODO: device_timer.wrap is too broad here — it also includes
            # replay_prepare time. Move timing into the piecewise cuda graph
            # runner to capture only the model.forward part.
            ctx = (
                self.device_timer.wrap(metadata={"category": "extend"})
                if self.device_timer
                else contextlib.nullcontext()
            )
            with ctx:
                ret = self.piecewise_cuda_graph_runner.replay(forward_batch, **kwargs)
            return (ret, can_run_graph)

        # Launch model forward
        if not skip_attn_backend_init:
            if hasattr(self.model, "prepare_forward_batch"):
                # Prepare model-specific attention metadata before planning,
                # e.g. Moss-VL's prefill cross-attention custom mask.
                self.model.prepare_forward_batch(forward_batch)
            self.attn_backend.init_forward_metadata(forward_batch)

        ctx = (
            self.device_timer.wrap(metadata={"category": "extend"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            ret = self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )
        return (ret, can_run_graph)

    # idle forward 用于 DP attention 等需要所有 rank 同步参与的场景。
    # 即使某个 rank 当前没有真实 token，也要维护 metadata 或清理旧 metadata，避免通信侧等待失败。
    def forward_idle(
        self, forward_batch: ForwardBatch, pp_proxy_tensors=None
    ) -> Union[LogitsProcessorOutput, PPProxyTensors]:
        # In DP Attention, IDLE batches may be padded (batch_size > 0) for MLP
        # sync. Reinit metadata for the padded case so attention kernels see
        # the right batch_size (e.g. DSA Indexer). For the unpadded case
        # (batch_size == 0) explicitly drop any stale forward_metadata left
        # over from the previous forward — without this, attention layers
        # called from the idle path can re-read a prior batch's req_pool
        # indices and trigger SWA mapping use-after-free.
        if forward_batch.batch_size > 0:
            self.attn_backend.init_forward_metadata(forward_batch)
        else:
            self.attn_backend.forward_metadata = None

        kwargs = {}
        if self.support_pp:
            kwargs["pp_proxy_tensors"] = pp_proxy_tensors
        ctx = (
            self.device_timer.wrap(metadata={"category": "idle"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            return self.model.forward(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                **kwargs,
            )

    # split prefill 的模型执行段。
    # 它会根据 split_index 和 split_forward_count 计算当前片段范围，然后调用模型的 forward_split_prefill。
    # 执行后推进 split_index，供下一次调用继续处理剩余 prompt。
    def forward_split_prefill(
        self,
        forward_batch: ForwardBatch,
        reinit_attn_backend: bool = False,
        forward_count: int = 1,
    ) -> LogitsProcessorOutput:
        if forward_batch.split_index == 0 or reinit_attn_backend:
            self.attn_backend.init_forward_metadata(forward_batch)
        next_split_index = min(
            forward_batch.split_index + forward_count,
            self.model_config.num_hidden_layers,
        )
        ctx = (
            self.device_timer.wrap(metadata={"category": "split_prefill"})
            if self.device_timer
            else contextlib.nullcontext()
        )
        with ctx:
            ret = self.model.forward_split_prefill(
                forward_batch.input_ids,
                forward_batch.positions,
                forward_batch,
                (forward_batch.split_index, next_split_index),
            )
        forward_batch.split_index = next_split_index
        return ret

    # ModelRunner 对外统一 forward 入口。
    # 它负责包裹 profiling、canary、expert distribution recorder、EPLB 和错误恢复等横切逻辑。
    # 真正根据 ForwardMode 选择执行路径的代码在 _forward_raw 中。
    def forward(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        self.forward_pass_id += 1

        # Try msprob debugger
        if self.msprobe_debugger is not None:
            rank_id = (
                self.gpu_id if self.dp_size is not None and self.dp_size > 1 else None
            )
            self.msprobe_debugger.start(model=self.model, rank_id=rank_id)

        # Step span
        step_span_ctx = (
            torch.profiler.record_function(_build_step_span_name(forward_batch))
            if torch.autograd._profiler_enabled()
            else contextlib.nullcontext()
        )

        canary_ctx = (
            context_tuple(
                c.with_ops_outside_graph(
                    single_forward_indices=[0],
                    maybe_inaccurate_forward_batch=forward_batch,
                ),
                c.with_active_single_forward_manager(0),
            )
            if not self.is_draft_worker and ((c := self.canary_manager) is not None)
            else contextlib.nullcontext()
        )

        with (
            canary_ctx,
            step_span_ctx,
            get_global_expert_distribution_recorder().with_forward_pass(
                self.forward_pass_id,
                forward_batch,
            ) as recorder_outputs,
        ):
            output = self._forward_raw(
                forward_batch,
                skip_attn_backend_init,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
            if self.enable_elastic_ep:
                output = self._maybe_rebalance_after_rank_fault(
                    output,
                    forward_batch,
                    skip_attn_backend_init,
                    pp_proxy_tensors,
                    reinit_attn_backend,
                    split_forward_count,
                )
        output.expert_distribution_metrics = recorder_outputs.get("metrics")

        no_copy_to_cpu = not self.server_args.disable_overlap_schedule
        if (experts_capturer := get_global_experts_capturer()) is not None:
            output.routed_experts_output = experts_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if (indexer_capturer := get_global_indexer_capturer()) is not None:
            output.indexer_topk_output = indexer_capturer.on_forward_end(
                forward_batch=forward_batch,
                can_run_graph=output.can_run_graph,
                cuda_graph_batch=getattr(self.graph_runner, "bs", None),
                no_copy_to_cpu=no_copy_to_cpu,
            )

        if self.eplb_manager is not None:
            self.eplb_manager.on_forward_pass_end()

        if dumper.may_enable:
            dumper.step()

        if self.msprobe_debugger is not None:
            self.msprobe_debugger.stop()
            self.msprobe_debugger.step()

        if self.server_args.elastic_ep_backend is not None:
            self.maybe_recover_ep_ranks()

        return output

    # ModelRunner 的核心 forward 分发器。
    # 这里先建立 ForwardContext，让模型内部 attention 能拿到当前 backend。
    # 然后根据 batch 形状判断是否可走 graph replay；如果不能 replay，就根据 ForwardMode 分发到 decode、extend、split prefill 或 idle。
    def _forward_raw(
        self,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool = False,
        split_forward_count: int = 1,
    ) -> ModelRunnerOutput:
        # Honor an outer-published context (spec workers wrap each per-step
        # draft forward with the i-th child backend); otherwise publish this
        # runner's own attn_backend for the forward.
        if has_forward_context():
            ctx_mgr = contextlib.nullcontext()
        else:
            ctx_mgr = forward_context(ForwardContext(attn_backend=self.attn_backend))
        with ctx_mgr:
            mode_check = (
                forward_batch.forward_mode.is_cpu_graph
                if self.device == "cpu"
                else forward_batch.forward_mode.is_cuda_graph
            )
            can_run_graph = bool(
                mode_check()
                and self.graph_runner
                # 判断当前 ForwardBatch 是否能被已捕获 graph 覆盖。
                # 如果 batch size、token 数、padding 模式等条件匹配，就可以走 replay 快路径。
                and self.graph_runner.can_run(forward_batch)
            )

            # Hisparse coordinator — backends now read it from self.model_runner.
            if (
                forward_batch.forward_mode.is_decode()
                and self.hisparse_coordinator is not None
            ):
                self.hisparse_coordinator.wait_for_pending_backup()
                self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

            if self.is_hybrid_swa:
                self.token_to_kv_pool.invalidate_loc_cache()

            # Replay cuda graph if applicable
            if can_run_graph:
                ret = self.graph_runner.replay(
                    forward_batch,
                    skip_attn_backend_init=skip_attn_backend_init,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
                return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

            # For MLP sync
            if forward_batch.global_num_tokens_cpu is not None:
                forward_batch.prepare_mlp_sync_batch(self)
            else:
                forward_batch.prepare_attn_tp_scatter_input(self)

            # Normalize num_token_non_padded to be local to this attention TP rank if needed.
            # The skip is scoped to DSACPLayerCommunicator-style CP (DSA, MLA): those
            # flavors already feed a zigzag-split rank-local layout whose token count
            # should not be further divided by attn_tp_size. MHA-arch prefill CP
            # (Qwen3/Qwen2 MoE) keeps the attn_tp-replicated layout and wants the
            # adjustment to run — see docs/design/prefill-cp-mla.md §Phase 5.
            if (
                forward_batch.num_token_non_padded is not None
                and forward_batch.global_num_tokens_gpu is not None
                and require_gathered_buffer(self.server_args)
                and not is_dsa_enable_prefill_cp()
                and not is_mla_prefill_cp_enabled()
            ):
                forward_batch.adjust_num_token_non_padded_for_attn_tp(
                    server_args=self.server_args,
                )

            # Hisparse coordinator — backends now read it from self.model_runner.
            if self.hisparse_coordinator is not None:
                self.hisparse_coordinator.num_real_reqs.fill_(forward_batch.batch_size)

            # Forward without cuda graph
            # decode 分支。
            # 每个请求通常只推进一个 token，是服务中最高频、最需要低延迟优化的路径。
            if forward_batch.forward_mode.is_decode():
                ret = self.forward_decode(
                    forward_batch,
                    skip_attn_backend_init=skip_attn_backend_init,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            # split prefill 分支。
            # 当前调用只处理长 prompt 的一个片段，处理完成后等待下一次 split 继续推进。
            elif forward_batch.forward_mode.is_split_prefill():
                ret = self.forward_split_prefill(
                    forward_batch,
                    reinit_attn_backend=reinit_attn_backend,
                    forward_count=split_forward_count,
                )
            # extend/prefill 分支。
            # 它处理 prompt 或新追加上下文，token 数多且形状动态，通常需要重新准备 attention metadata。
            elif forward_batch.forward_mode.is_extend(include_draft_extend_v2=True):
                ret, can_run_graph = self.forward_extend(
                    forward_batch,
                    skip_attn_backend_init=skip_attn_backend_init,
                    pp_proxy_tensors=pp_proxy_tensors,
                )
            # idle 分支。
            # 某些并行模式下，即使没有本地真实 token，也要执行空 forward 来保持 rank 间同步。
            elif forward_batch.forward_mode.is_idle():
                ret = self.forward_idle(
                    forward_batch, pp_proxy_tensors=pp_proxy_tensors
                )
            else:
                raise ValueError(f"Invalid forward mode: {forward_batch.forward_mode}")

            if (
                forward_batch.global_num_tokens_cpu is not None
                and self.pp_group.is_last_rank
            ):
                forward_batch.post_forward_mlp_sync_batch(ret)

            return ModelRunnerOutput(logits_output=ret, can_run_graph=can_run_graph)

    # 采样前的 logits 预处理。
    # 这里会应用 logit bias、softcap、grammar/mask、logprob 需要的归一化等逻辑。
    # 预处理后的 logits 才会交给 sampler 执行 temperature/top-p/top-k 等采样策略。
    def _preprocess_logits(
        self, logits_output: LogitsProcessorOutput, sampling_info: SamplingBatchInfo
    ):
        # NOTE: In overlap mode, the function update_regex_vocab_mask (in sample)
        #       was executed after we processed last batch's results.
        # Calculate logits bias and apply it to next_token_logits.

        sampling_info.update_regex_vocab_mask()
        sampling_info.apply_logits_bias(logits_output.next_token_logits)

        # Release the vocab_mask GPU tensor immediately after it has been applied
        # to the logits. In overlap scheduling, the sampling_info (and its
        # vocab_mask) can be kept alive by the delay_sample_func closure and
        # batch_record_buf until the next iteration, causing a steady VRAM leak
        # when structured output (grammar) is used.
        sampling_info.vocab_mask = None

    # 根据 logits 和 sampling_info 生成 next token。
    # decode 模式采样位置是一批请求的一维位置；prefill 模式通常取每个序列末尾 token 的 logits。
    # 采样后还会更新 ngram token table，保持后续 forward 的辅助状态一致。
    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Sample and compute logprobs and update logits_output.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output

        Returns:
            A list of next_token_ids
        """
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Sample the next tokens
        # 调用采样器。
        # sampler 会根据 sampling_info 执行 temperature、top-p、top-k、min-p、grammar mask、logprob 等策略。
        next_token_ids = self.sampler(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
            # For prefill, we only use the position of the last token.
            (
                forward_batch.positions
                if forward_batch.forward_mode.is_decode()
                else forward_batch.seq_lens - 1
            ),
        )
        self.maybe_update_ngram_token_table(next_token_ids, forward_batch)
        return next_token_ids

    # 只计算 logprob，不生成 next token。
    # 这个路径服务 return_logprob、prefill-only 或验证类请求，避免走不必要的 sampling。
    def compute_logprobs_only(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
    ) -> None:
        """
        Compute token_ids_logprobs without performing sampling.

        Optimized path for prefill-only requests that need token_ids_logprobs but don't
        require next token generation. Skips expensive sampling operations
        while still providing requested probability information.

        Args:
            logits_output: The logits output from the model forward
            forward_batch: The forward batch that generates logits_output
        """
        if not forward_batch.token_ids_logprobs:
            return

        # Preprocess logits (same as in sample method)
        self._preprocess_logits(logits_output, forward_batch.sampling_info)

        # Delegate to sampler for logprob-only computation
        # This populates logits_output with requested token probabilities
        self.sampler.compute_logprobs_only(
            logits_output,
            forward_batch.sampling_info,
            forward_batch.return_logprob,
            forward_batch.top_logprobs_nums,
            forward_batch.token_ids_logprobs,
        )

    # 把当前模型保存到远端位置。
    # 这通常用于把运行时模型权重导出给外部存储或其他实例复用。
    def save_remote_model(self, url: str):
        from sglang.srt.model_loader.loader import RemoteModelLoader

        logger.info(f"Saving model to {url}")
        RemoteModelLoader.save_model(self.model, self.model_config.model_path, url)

    # 保存当前 rank 负责的分片模型。
    # 在 TP/PP 场景中，每个 rank 只持有部分权重，因此保存时需要按 sharded 格式组织输出。
    def save_sharded_model(
        self, path: str, pattern: Optional[str] = None, max_size: Optional[int] = None
    ):
        from sglang.srt.model_loader.loader import ShardedStateLoader

        logger.info(
            f"Save sharded model to {path} with pattern {pattern} and max_size {max_size}"
        )
        ShardedStateLoader.save_model(self.model, path, pattern, max_size)

    # `check_weights` 负责校验运行时假设。
    # 这类检查通常用来提前发现权重、并行配置或后端能力不匹配的问题。
    def check_weights(self, action: str):
        return self._weight_checker.handle(action=action)

    # 通过 IPC 共享内存或句柄接收权重 tensor。
    # 这种方式避免大 tensor 在进程间重复拷贝，适合本机多进程权重热更新。
    def update_weights_from_ipc(self, recv_req):
        """Update weights from IPC for checkpoint-engine integration."""
        try:
            from sglang.srt.checkpoint_engine.checkpoint_engine_worker import (
                SGLangCheckpointEngineWorkerExtensionImpl,
            )

            # Create a worker extension that integrates with SGLang's model
            worker = SGLangCheckpointEngineWorkerExtensionImpl(self)
            worker.update_weights_from_ipc(recv_req.zmq_handles)
            return True, "IPC weight update completed successfully"
        except ImportError as e:
            return False, f"IPC weight update failed: ImportError {e}"
        except Exception as e:
            logger.error(f"IPC weight update failed: {e}")
            return False, str(e)

    # 预分配 symmetric memory pool。
    # symmetric memory 要求多个 rank 上的内存布局对齐，提前分配可以支持特定通信或 fused kernel。
    def prealloc_symmetric_memory_pool(self):
        # PyTorch mempools never de-fragment memory in OOM scenarios, so we need to pre-allocate a large chunk of memory to limit fragmentation.
        if (
            self.is_draft_worker
            or not self.server_args.enable_symm_mem
            or envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() <= 0
        ):
            return

        # Memory allocation is tied to a cuda stream, use the forward stream
        with torch.get_device_module(self.device).stream(self.forward_stream):
            logger.info(
                f"Pre-allocating symmetric memory pool with {envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get()} GiB"
            )
            with use_symmetric_memory(get_tp_group()):
                torch.empty(
                    (envs.SGLANG_SYMM_MEM_PREALLOC_GB_SIZE.get() * 1024 * 1024 * 1024,),
                    dtype=torch.uint8,
                    device=self.device,
                )

    # rank 故障后尝试重新平衡专家或运行时状态。
    # 这是 elastic/容错路径的一部分，用于让服务在部分 rank 异常后尽量恢复可用。
    def _maybe_rebalance_after_rank_fault(
        self,
        output: ModelRunnerOutput,
        forward_batch: ForwardBatch,
        skip_attn_backend_init: bool,
        pp_proxy_tensors: Optional[PPProxyTensors],
        reinit_attn_backend: bool,
        split_forward_count: int,
    ) -> ModelRunnerOutput:
        elastic_ep_state = ElasticEPStateManager.instance()
        if elastic_ep_state is not None and not elastic_ep_state.is_active_equal_last():
            elastic_ep_state.snapshot_active_to_last()
            elastic_ep_state.sync_active_to_cpu()
            logging.info("EPLB due to rank faults")
            gen = self.eplb_manager.rebalance()
            while True:
                try:
                    next(gen)
                except StopIteration:
                    break
            output = self._forward_raw(
                forward_batch,
                skip_attn_backend_init,
                pp_proxy_tensors,
                reinit_attn_backend,
                split_forward_count,
            )
        return output


# 直接把给定 named_tensors 写入模型。
# 这是权重更新底层工具函数，绕过高级 loader，逐个参数调用模型的加载逻辑。
def _model_load_weights_direct(model, named_tensors: List[Tuple[str, torch.Tensor]]):
    params_dict = dict(model.named_parameters())
    for name, tensor in named_tensors:
        default_weight_loader(params_dict[name], tensor)


# 从序列化或封装对象中取出当前 rank 需要的 tensor。
# 如果 tensor 是按 rank 分片保存的，这里会选择对应 tp_rank 的分片并移动到目标 device。
def _unwrap_tensor(tensor, tp_rank, device):
    if isinstance(tensor, LocalSerializedTensor):
        tensor = tensor.get(tp_rank)
    return tensor.to(device)


# 根据 ForwardBatch 构造 tracing/profiling span 名称。
# 名称中包含 forward mode 等信息，便于在性能分析工具中区分 decode、extend、verify 等阶段。
def _build_step_span_name(forward_batch: ForwardBatch) -> str:
    """Build a profile-trace span name for one forward step."""
    mode = forward_batch.forward_mode
    bs = forward_batch.batch_size
    if mode == ForwardMode.EXTEND:
        ext_toks = forward_batch.extend_num_tokens or 0
        return f"step[EXTEND bs={bs} toks={ext_toks}]"
    return f"step[{mode.name} bs={bs}]"


@dataclass
# 用于权重更新流程的本地序列化 tensor 描述。
# 它把 tensor 的存储位置、形状和 dtype 信息包装起来，方便跨接口传递。
class LocalSerializedTensor:
    """torch.Tensor that gets serialized by MultiprocessingSerializer (which only serializes a pointer and not the data).
    The i-th element in the list corresponds to i-th rank's GPU."""

    values: List[bytes]

    # 读取 LocalSerializedTensor 中指定 rank 的 tensor。
    # 权重更新流程可以通过该方法为不同 TP rank 取出对应参数分片。
    def get(self, rank: int):
        return MultiprocessingSerializer.deserialize(self.values[rank])
