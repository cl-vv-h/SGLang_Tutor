# sp_context_parallel_demo.py

"""
教学目标：
    更接近工业实现地理解 Sequence / Context Parallelism。

和简单教学版的区别：

    简单版：
        每个 rank 都 all_gather 全局 K/V。
        然后 local Q attend full K/V。

        问题：
            1. 每个 rank 都 materialize 全量 KV；
            2. 对长上下文非常浪费显存和通信；
            3. causal 场景下前面的 rank 拿未来 KV 是冗余的。

    改进版：
        每个 rank 只保存本地 KV shard。
        Attention 时不 all_gather KV。
        而是：
            1. 每个 rank 用本地 K/V 计算局部 score；
            2. 用 all_reduce(max) 得到全局 softmax max；
            3. 用 all_reduce(sum) 得到全局 softmax denominator；
            4. 每个 rank 计算本地 value 加权贡献；
            5. 用 all_reduce(sum) 合并 context。

核心思想：
    Attention(Q, K, V) = softmax(QK^T)V

    如果 K/V 按 sequence 分片：
        rank0: K0, V0
        rank1: K1, V1

    那么对于同一个 Q：

        scores0 = Q @ K0^T
        scores1 = Q @ K1^T

    softmax 必须在 [scores0, scores1] 这个全局范围上做。

    所以不能简单地各 rank 本地 softmax 后再相加。
    正确做法是 distributed softmax：

        global_max = max(max(scores0), max(scores1))
        global_sum = sum(exp(scores_i - global_max))
        context = sum(exp(scores_i - global_max) @ V_i) / global_sum

    这就是本代码最核心的部分。
"""

import os
import math
import argparse
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ============================================================
# 1. 分布式初始化
# ============================================================

def init_distributed(use_cpu: bool = False):
    """
    初始化分布式环境。

    torchrun 会提供：
        RANK
        WORLD_SIZE
        LOCAL_RANK

    在本代码中：
        world_size = context parallel size
                   = sequence 被切成几份
    """

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if use_cpu:
        backend = "gloo"
        device = torch.device("cpu")
    else:
        backend = "nccl"
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    return rank, world_size, local_rank, device


def cleanup_distributed():
    dist.destroy_process_group()


# ============================================================
# 2. sequence 切分工具
# ============================================================

def split_sequence_for_rank(
    full_input_ids: torch.Tensor,
    rank: int,
    world_size: int,
) -> Tuple[torch.Tensor, int, int]:
    """
    把完整输入序列按 sequence 维度切分。

    full_input_ids.shape = [B, S]

    假设：
        S = 8
        world_size = 2

    rank 0:
        token 0~3

    rank 1:
        token 4~7

    返回：
        local_input_ids:
            当前 rank 负责的 token ids。

        local_start:
            当前 shard 在全局序列中的起始位置。

        local_end:
            当前 shard 在全局序列中的结束位置。
    """

    B, S = full_input_ids.shape

    assert S % world_size == 0, (
        "教学版要求 seq_len 能被 world_size 整除"
    )

    local_S = S // world_size

    local_start = rank * local_S
    local_end = local_start + local_S

    local_input_ids = full_input_ids[:, local_start:local_end].contiguous()

    return local_input_ids, local_start, local_end


# ============================================================
# 3. 分布式 softmax attention
# ============================================================

def distributed_softmax_attention(
    q: torch.Tensor,
    local_k: torch.Tensor,
    local_v: torch.Tensor,
    q_start_pos: int,
    k_start_pos: int,
    causal: bool,
) -> torch.Tensor:
    """
    使用分布式 softmax 计算 attention。

    这是本代码最核心的函数。

    输入：
        q:
            当前要计算的 query。
            shape = [B, H, Q, D]

        local_k:
            当前 rank 保存的 K shard。
            shape = [B, H, K_local, D]

        local_v:
            当前 rank 保存的 V shard。
            shape = [B, H, K_local, D]

        q_start_pos:
            q 的第一个 token 在全局序列中的位置。

        k_start_pos:
            local_k 的第一个 token 在全局序列中的位置。
            prefill 时一般是当前 rank 的 local_start。
            decode 时如果 KV 采用动态 block/page 分配，位置可能更复杂。
            本教学代码中 prefill 用 contiguous shard，decode 不再依赖 mask。

        causal:
            是否使用 causal mask。
            prefill 阶段需要 True。
            decode 最新 token attend 历史时可以 False，因为 cache 中都是历史和当前 token。

    输出：
        context:
            shape = [B, H, Q, D]

    关键点：
        这个函数没有 all_gather K/V。

        每个 rank 只基于自己的 local_k/local_v 计算局部贡献。
        然后通过 all_reduce 合并 softmax 所需的全局 max、sum 和 context。
    """

    B, num_heads, q_len, head_dim = q.shape
    k_len = local_k.size(2)
    device = q.device
    dtype = q.dtype

    # ------------------------------------------------------------
    # 1. 当前 rank 计算局部 attention scores
    # ------------------------------------------------------------
    #
    # q.shape:
    #   [B, H, Q, D]
    #
    # local_k.transpose(-2, -1).shape:
    #   [B, H, D, K_local]
    #
    # local_scores.shape:
    #   [B, H, Q, K_local]
    if k_len > 0:
        local_scores = torch.matmul(q, local_k.transpose(-2, -1))
        local_scores = local_scores / math.sqrt(head_dim)

        # --------------------------------------------------------
        # 2. causal mask
        # --------------------------------------------------------
        #
        # prefill 阶段:
        #   当前 rank 的 q 可能对应全局位置 [4,5,6,7]
        #   当前 rank 的 k 可能对应全局位置 [0,1,2,3]
        #
        #   所以 mask 不能简单用 tril(local_S, local_S)，
        #   必须基于全局位置判断：
        #
        #       key_pos <= query_pos
        #
        if causal:
            query_pos = torch.arange(
                q_start_pos,
                q_start_pos + q_len,
                device=device,
            ).unsqueeze(-1)

            key_pos = torch.arange(
                k_start_pos,
                k_start_pos + k_len,
                device=device,
            ).unsqueeze(0)

            # mask.shape = [Q, K_local]
            mask = key_pos <= query_pos

            # broadcast 到 [B, H, Q, K_local]
            local_scores = local_scores.masked_fill(
                mask.unsqueeze(0).unsqueeze(0) == 0,
                float("-inf"),
            )

        # 当前 rank 对每个 query 的局部最大值。
        #
        # local_max.shape = [B, H, Q, 1]
        local_max = torch.max(local_scores, dim=-1, keepdim=True).values

    else:
        # 某些极端情况下，一个 rank 可能暂时没有 KV。
        # 为了让 all_reduce 能正常执行，构造空贡献。
        local_scores = None
        local_max = torch.full(
            (B, num_heads, q_len, 1),
            fill_value=float("-inf"),
            dtype=dtype,
            device=device,
        )

    # ------------------------------------------------------------
    # 3. all_reduce max 得到全局 softmax max
    # ------------------------------------------------------------
    #
    # softmax 的数值稳定写法：
    #
    #   softmax(scores) = exp(scores - max(scores)) / sum(exp(scores - max(scores)))
    #
    # 现在 scores 被分散在不同 rank 上，
    # 所以需要所有 rank 一起求 global_max。
    global_max = local_max.clone()
    dist.all_reduce(global_max, op=dist.ReduceOp.MAX)

    # ------------------------------------------------------------
    # 4. 计算局部 exp 和局部 denominator
    # ------------------------------------------------------------
    if k_len > 0:
        local_exp = torch.exp(local_scores - global_max)

        # 被 mask 的 -inf 位置 exp 后是 0。
        #
        # local_sum.shape = [B, H, Q, 1]
        local_sum = torch.sum(local_exp, dim=-1, keepdim=True)

        # 局部 value 加权和：
        #
        # local_exp.shape = [B, H, Q, K_local]
        # local_v.shape   = [B, H, K_local, D]
        #
        # local_out.shape = [B, H, Q, D]
        local_out = torch.matmul(local_exp, local_v)

    else:
        local_sum = torch.zeros(
            (B, num_heads, q_len, 1),
            dtype=dtype,
            device=device,
        )

        local_out = torch.zeros(
            (B, num_heads, q_len, head_dim),
            dtype=dtype,
            device=device,
        )

    # ------------------------------------------------------------
    # 5. all_reduce sum 得到全局 denominator 和全局 numerator
    # ------------------------------------------------------------
    global_sum = local_sum.clone()
    dist.all_reduce(global_sum, op=dist.ReduceOp.SUM)

    global_out = local_out.clone()
    dist.all_reduce(global_out, op=dist.ReduceOp.SUM)

    # ------------------------------------------------------------
    # 6. 得到最终 context
    # ------------------------------------------------------------
    #
    # context = sum(exp(scores_i - global_max) @ V_i) / global_sum
    #
    # 这是严格等价于对全局 K/V 做 softmax attention 的。
    context = global_out / global_sum.clamp_min(1e-9)

    return context


# ============================================================
# 4. Sharded KV Cache
# ============================================================

class ShardedKVCache:
    """
    一个简化版的 sequence-sharded KV Cache。

    每个 rank 只保存自己负责的 KV shard。

    和前面简单代码不同：
        不会把所有 rank 的 KV gather 到每个 rank。

    这里为了教学，使用预分配 tensor：

        key_cache[layer].shape =
            [B, H, max_local_tokens, D]

    每个 rank 自己维护每一层的 local cache length。

    真实工业实现中通常会更复杂：
        1. Paged KV Cache；
        2. block table；
        3. prefix cache；
        4. request-level cache manager；
        5. dynamic batching；
        6. KV eviction / swap。
    """

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        max_local_tokens: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.max_local_tokens = max_local_tokens
        self.dtype = dtype
        self.device = device

        self.key_cache = torch.empty(
            num_layers,
            batch_size,
            num_heads,
            max_local_tokens,
            head_dim,
            dtype=dtype,
            device=device,
        )

        self.value_cache = torch.empty(
            num_layers,
            batch_size,
            num_heads,
            max_local_tokens,
            head_dim,
            dtype=dtype,
            device=device,
        )

        # 每一层当前本 rank 已经缓存了多少 token。
        self.cache_lens = [0 for _ in range(num_layers)]

    def write_prefill(
        self,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """
        写入 prefill 阶段的本地 K/V shard。

        k/v.shape = [B, H, local_S, D]

        每个 rank 只写自己负责的 local_S 个 token。
        """

        B, H, T, D = k.shape

        assert T <= self.max_local_tokens

        self.key_cache[layer_id, :, :, :T, :] = k
        self.value_cache[layer_id, :, :, :T, :] = v

        self.cache_lens[layer_id] = T

    def append_decode_token(
        self,
        layer_id: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """
        decode 阶段追加一个新 token 的 K/V。

        k/v.shape = [B, H, 1, D]

        注意：
            只有当前 token 的 owner rank 会调用这个函数。
            非 owner rank 不写入。
        """

        pos = self.cache_lens[layer_id]

        if pos >= self.max_local_tokens:
            raise RuntimeError("Local KV cache is full")

        self.key_cache[layer_id, :, :, pos:pos + 1, :] = k
        self.value_cache[layer_id, :, :, pos:pos + 1, :] = v

        self.cache_lens[layer_id] += 1

    def get_kv(
        self,
        layer_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取当前 rank 在某一层保存的全部本地 KV。

        返回：
            k.shape = [B, H, local_cache_len, D]
            v.shape = [B, H, local_cache_len, D]
        """

        T = self.cache_lens[layer_id]

        k = self.key_cache[layer_id, :, :, :T, :]
        v = self.value_cache[layer_id, :, :, :T, :]

        return k, v


# ============================================================
# 5. Context Parallel Attention
# ============================================================

class ContextParallelSelfAttention(nn.Module):
    """
    更接近工业实现的 Context Parallel Attention。

    两种路径：

    1. prefill:
        每个 rank 持有一段 local sequence。
        每个 rank 只保存本地 K/V。
        为了计算每个 rank 的 local query output：
            逐个 query owner rank 广播 Q shard；
            所有 rank 用本地 K/V 算局部 attention；
            all_reduce 合并 context；
            owner rank 保存自己的 local output。

        这样避免 all_gather 全局 K/V。

    2. decode:
        每次只有一个最新 token。
        当前 token 的 Q 对所有 rank 都相同。
        每个 rank 用自己保存的 KV shard 计算局部贡献。
        all_reduce 后每个 rank 都得到最新 token 的 context。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        layer_id: int,
        rank: int,
        world_size: int,
    ):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_id = layer_id
        self.rank = rank
        self.world_size = world_size

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, S, H] -> [B, num_heads, S, head_dim]
        """

        B, S, H = x.shape

        x = x.view(B, S, self.num_heads, self.head_dim)

        x = x.transpose(1, 2).contiguous()

        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [B, num_heads, S, head_dim] -> [B, S, H]
        """

        B, H, S, D = x.shape

        x = x.transpose(1, 2).contiguous()

        x = x.view(B, S, H * D)

        return x

    def prefill(
        self,
        local_x: torch.Tensor,
        local_start: int,
        local_S: int,
        cache: ShardedKVCache,
    ) -> torch.Tensor:
        """
        prefill 阶段。

        local_x.shape = [B, local_S, hidden_size]

        每个 rank 只拥有一段 local query。
        但为了计算某个 owner rank 的 Q 对全局 K/V 的 attention，
        所有 rank 都需要参与计算。

        具体流程：
            1. 每个 rank 计算自己的 local Q/K/V；
            2. 每个 rank 把自己的 K/V 写进本地 KV cache；
            3. 对 owner_rank = 0..world_size-1 循环：
                a. owner_rank 广播自己的 q_local；
                b. 所有 rank 用自己的 local K/V 计算局部 attention；
                c. all_reduce 合并输出；
                d. owner_rank 保存对应输出。
            4. 当前 rank 返回自己的 local output。

        这样做的效果是：
            不需要任何 rank 保存全局 KV。
        """

        B, _, H = local_x.shape
        device = local_x.device

        # ------------------------------------------------------------
        # 1. 本地计算 Q/K/V
        # ------------------------------------------------------------
        q_local = self._split_heads(self.q_proj(local_x))
        k_local = self._split_heads(self.k_proj(local_x))
        v_local = self._split_heads(self.v_proj(local_x))

        # ------------------------------------------------------------
        # 2. 本地 K/V 写入 sharded KV cache
        # ------------------------------------------------------------
        cache.write_prefill(
            layer_id=self.layer_id,
            k=k_local,
            v=v_local,
        )

        # 当前 rank 最终要得到自己的 local context。
        local_context = torch.empty_like(q_local)

        # ------------------------------------------------------------
        # 3. 逐个 query owner 计算分布式 attention
        # ------------------------------------------------------------
        #
        # 为什么要 broadcast Q？
        #
        # 因为 distributed softmax attention 要求：
        #   所有 rank 针对同一批 Q，分别用自己的 K/V shard
        #   计算局部贡献，然后 all_reduce 合并。
        #
        # 如果 rank0 用 Q0，rank1 用 Q1 同时 all_reduce，
        # 那语义是错的。
        #
        # 所以我们让 owner 的 Q shard 广播给所有 rank，
        # 大家一起为这个 Q shard 计算 attention。
        for owner_rank in range(self.world_size):
            if self.rank == owner_rank:
                q_owner = q_local.contiguous()
            else:
                q_owner = torch.empty_like(q_local)

            # 广播 owner_rank 的 Q shard。
            dist.broadcast(q_owner, src=owner_rank)

            q_start = owner_rank * local_S

            # 所有 rank 用自己的 local K/V 参与计算。
            owner_context = distributed_softmax_attention(
                q=q_owner,
                local_k=k_local,
                local_v=v_local,
                q_start_pos=q_start,
                k_start_pos=local_start,
                causal=True,
            )

            # 只有 owner rank 保留这个输出。
            if self.rank == owner_rank:
                local_context.copy_(owner_context)

        # ------------------------------------------------------------
        # 4. 合并 heads + 输出投影
        # ------------------------------------------------------------
        local_context = self._merge_heads(local_context)

        out = self.o_proj(local_context)

        return out

    def decode(
        self,
        x_token: torch.Tensor,
        global_pos: int,
        cache: ShardedKVCache,
    ) -> torch.Tensor:
        """
        decode 阶段。

        x_token.shape = [B, 1, hidden_size]

        此时只需要计算最新 token 的输出。

        流程：
            1. 所有 rank 都有最新 token 的 hidden；
            2. 所有 rank 都计算 q_new/k_new/v_new；
            3. 根据 token_owner 判断哪个 rank 保存这个 token 的 K/V；
            4. 每个 rank 用自己本地 cache 中的 K/V 计算局部 attention；
            5. all_reduce 合并得到最新 token 的 context。

        注意：
            decode 阶段不需要每个 rank 重新计算历史 token 输出。
            历史 rank 的作用是：
                持有历史 KV shard，并为最新 token 提供局部 attention 贡献。
        """

        # ------------------------------------------------------------
        # 1. 计算最新 token 的 Q/K/V
        # ------------------------------------------------------------
        q = self._split_heads(self.q_proj(x_token))
        k_new = self._split_heads(self.k_proj(x_token))
        v_new = self._split_heads(self.v_proj(x_token))

        # ------------------------------------------------------------
        # 2. 决定最新 token 的 KV 存在哪个 rank
        # ------------------------------------------------------------
        #
        # 教学版用 round-robin 分配 generated token 的 KV。
        #
        # 真实系统通常由 KV cache manager / block table 决定。
        token_owner = global_pos % self.world_size

        if self.rank == token_owner:
            cache.append_decode_token(
                layer_id=self.layer_id,
                k=k_new,
                v=v_new,
            )

        # ------------------------------------------------------------
        # 3. 获取本 rank 本地保存的全部 KV shard
        # ------------------------------------------------------------
        local_k, local_v = cache.get_kv(self.layer_id)

        # ------------------------------------------------------------
        # 4. 最新 token attend 分布式 KV cache
        # ------------------------------------------------------------
        #
        # decode 阶段：
        #   local_k/local_v 中保存的都是历史 token 或当前 token。
        #   对最新 token 来说，它们都不是未来 token。
        #
        # 所以 causal=False。
        context = distributed_softmax_attention(
            q=q,
            local_k=local_k,
            local_v=local_v,
            q_start_pos=global_pos,
            k_start_pos=0,
            causal=False,
        )

        context = self._merge_heads(context)

        out = self.o_proj(context)

        return out


# ============================================================
# 6. Transformer Block
# ============================================================

class FeedForward(nn.Module):
    """
    FFN 是逐 token 计算的。

    在 sequence/context parallel 中：
        prefill 阶段，每个 rank 只对自己的 local token 做 FFN；
        decode 阶段，只对最新 token 做 FFN。
    """

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()

        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = F.gelu(x)
        x = self.down_proj(x)
        return x


class ContextParallelBlock(nn.Module):
    """
    一个 context-parallel Transformer Block。

    prefill:
        输入 local_x，输出 local_x。
        每个 rank 只持有自己负责的 sequence shard。

    decode:
        输入最新 token hidden，输出最新 token hidden。
        所有 rank 都参与 attention，因为每个 rank 持有部分 KV cache。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        layer_id: int,
        rank: int,
        world_size: int,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)

        self.attn = ContextParallelSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            layer_id=layer_id,
            rank=rank,
            world_size=world_size,
        )

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size)

    def prefill(
        self,
        local_x: torch.Tensor,
        local_start: int,
        local_S: int,
        cache: ShardedKVCache,
    ) -> torch.Tensor:
        attn_out = self.attn.prefill(
            local_x=self.norm1(local_x),
            local_start=local_start,
            local_S=local_S,
            cache=cache,
        )

        local_x = local_x + attn_out

        ffn_out = self.ffn(self.norm2(local_x))

        local_x = local_x + ffn_out

        return local_x

    def decode(
        self,
        x_token: torch.Tensor,
        global_pos: int,
        cache: ShardedKVCache,
    ) -> torch.Tensor:
        attn_out = self.attn.decode(
            x_token=self.norm1(x_token),
            global_pos=global_pos,
            cache=cache,
        )

        x_token = x_token + attn_out

        ffn_out = self.ffn(self.norm2(x_token))

        x_token = x_token + ffn_out

        return x_token


# ============================================================
# 7. 完整模型
# ============================================================

class TinyContextParallelLM(nn.Module):
    """
    一个更接近工业思路的 context-parallel decoder-only LM。

    prefill:
        每个 rank 处理 local sequence shard，并建立本地 KV cache。

    decode:
        所有 rank 共同处理最新 token；
        KV cache 按 sequence/token 分散保存在不同 rank；
        attention 通过分布式 softmax 合并结果。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        rank: int,
        world_size: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.world_size = world_size

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.blocks = nn.ModuleList([
            ContextParallelBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                layer_id=i,
                rank=rank,
                world_size=world_size,
            )
            for i in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def embed_local(
        self,
        local_input_ids: torch.Tensor,
        local_start: int,
    ) -> torch.Tensor:
        """
        prefill 阶段对本地 sequence shard 做 embedding。

        position id 必须使用全局位置。
        """

        B, local_S = local_input_ids.shape
        device = local_input_ids.device

        position_ids = torch.arange(
            local_start,
            local_start + local_S,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(B, local_S)

        x = self.token_embedding(local_input_ids)
        x = x + self.position_embedding(position_ids)

        return x

    def embed_token(
        self,
        token_id: torch.Tensor,
        global_pos: int,
    ) -> torch.Tensor:
        """
        decode 阶段对最新 token 做 embedding。

        token_id.shape = [B, 1]
        """

        B, _ = token_id.shape
        device = token_id.device

        position_ids = torch.full(
            (B, 1),
            fill_value=global_pos,
            dtype=torch.long,
            device=device,
        )

        x = self.token_embedding(token_id)
        x = x + self.position_embedding(position_ids)

        return x

    def prefill(
        self,
        local_input_ids: torch.Tensor,
        local_start: int,
        local_S: int,
        cache: ShardedKVCache,
    ) -> torch.Tensor:
        """
        prefill 阶段。

        返回当前 rank 的 local logits。
        只有包含最后一个 token 的 rank 的最后位置 logits
        才能用于预测第一个新 token。
        """

        x = self.embed_local(local_input_ids, local_start)

        for block in self.blocks:
            x = block.prefill(
                local_x=x,
                local_start=local_start,
                local_S=local_S,
                cache=cache,
            )

        x = self.final_norm(x)

        local_logits = self.lm_head(x)

        return local_logits

    def decode(
        self,
        token_id: torch.Tensor,
        global_pos: int,
        cache: ShardedKVCache,
    ) -> torch.Tensor:
        """
        decode 阶段。

        输入最新 token，输出这个 token 位置的 logits。

        注意：
            每个 rank 都会得到相同的最新 token hidden/logits。
            因为 attention context 通过 all_reduce 合并。
        """

        x = self.embed_token(token_id, global_pos)

        for block in self.blocks:
            x = block.decode(
                x_token=x,
                global_pos=global_pos,
                cache=cache,
            )

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


# ============================================================
# 8. 自回归生成
# ============================================================

@torch.no_grad()
def generate_with_context_parallel(
    model: TinyContextParallelLM,
    full_input_ids: torch.Tensor,
    rank: int,
    world_size: int,
    device: torch.device,
    max_new_tokens: int,
):
    """
    使用 context-parallel KV cache 做自回归生成。

    流程：

        1. prefill:
            sequence 被切成多个 shard。
            每个 rank 只处理自己的 shard。
            KV cache 也只保存在本 rank。

        2. 第一个 next token:
            由包含 prompt 最后 token 的 rank 产生。
            然后 broadcast 给所有 rank。

        3. decode:
            所有 rank 都拿到最新 token；
            每个 rank 只用本地 KV shard 计算 attention 局部贡献；
            all_reduce 合并；
            rank0 采样下一个 token；
            broadcast 给所有 rank。
    """

    B, S = full_input_ids.shape

    local_input_ids, local_start, local_end = split_sequence_for_rank(
        full_input_ids=full_input_ids,
        rank=rank,
        world_size=world_size,
    )

    local_S = local_input_ids.size(1)

    # 为了教学简单，每个 rank 最多存 max_seq_len 个本地 KV。
    # 工业实现通常使用 paged KV blocks，而不是这么粗暴地预分配。
    cache = ShardedKVCache(
        num_layers=model.num_layers,
        batch_size=B,
        num_heads=model.num_heads,
        head_dim=model.hidden_size // model.num_heads,
        max_local_tokens=model.max_seq_len,
        dtype=torch.float32,
        device=device,
    )

    print(
        f"[rank {rank}] prefill local range=[{local_start},{local_end}), "
        f"local_input_ids={local_input_ids.tolist()}",
        flush=True,
    )

    # ------------------------------------------------------------
    # 1. Prefill
    # ------------------------------------------------------------
    local_logits = model.prefill(
        local_input_ids=local_input_ids,
        local_start=local_start,
        local_S=local_S,
        cache=cache,
    )

    # 包含 prompt 最后一个 token 的 rank。
    last_prompt_rank = world_size - 1

    if rank == last_prompt_rank:
        next_token = torch.argmax(
            local_logits[:, -1, :],
            dim=-1,
            keepdim=True,
        )
    else:
        next_token = torch.empty(
            B,
            1,
            dtype=torch.long,
            device=device,
        )

    # 广播第一个生成 token。
    dist.broadcast(next_token, src=last_prompt_rank)

    generated_tokens: List[int] = []

    if rank == 0:
        generated_tokens.append(int(next_token[0, 0].item()))
        print(
            f"[rank 0] first next_token={generated_tokens[-1]}",
            flush=True,
        )

    # ------------------------------------------------------------
    # 2. Decode loop
    # ------------------------------------------------------------
    #
    # 已经通过 prefill logits 得到了第一个 next_token。
    # 接下来把这个 token 输入 decode，预测下一个 token。
    current_token = next_token

    current_global_pos = S

    for step in range(max_new_tokens - 1):
        logits = model.decode(
            token_id=current_token,
            global_pos=current_global_pos,
            cache=cache,
        )

        # 这里为了简单，让 rank0 负责采样。
        if rank == 0:
            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True,
            )
        else:
            next_token = torch.empty(
                B,
                1,
                dtype=torch.long,
                device=device,
            )

        dist.broadcast(next_token, src=0)

        if rank == 0:
            generated_tokens.append(int(next_token[0, 0].item()))
            print(
                f"[rank 0] decode step={step}, "
                f"next_token={generated_tokens[-1]}",
                flush=True,
            )

        current_token = next_token
        current_global_pos += 1

    return generated_tokens


# ============================================================
# 9. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=5)
    args = parser.parse_args()

    rank, world_size, local_rank, device = init_distributed(args.cpu)

    vocab_size = 100
    hidden_size = 128
    intermediate_size = 512
    num_heads = 8
    num_layers = 2
    max_seq_len = 64

    seq_len = 8
    batch_size = 1

    assert seq_len % world_size == 0

    torch.manual_seed(1234)

    model = TinyContextParallelLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        rank=rank,
        world_size=world_size,
    ).to(device)

    model.eval()

    full_input_ids = torch.tensor(
        [[10, 20, 30, 40, 50, 60, 70, 80]],
        dtype=torch.long,
        device=device,
    )

    if rank == 0:
        print(
            f"Context Parallel world_size={world_size}, "
            f"full_input_ids={full_input_ids.tolist()}",
            flush=True,
        )

    generated = generate_with_context_parallel(
        model=model,
        full_input_ids=full_input_ids,
        rank=rank,
        world_size=world_size,
        device=device,
        max_new_tokens=args.max_new_tokens,
    )

    if rank == 0:
        print("\n========== Generated Tokens ==========")
        print(generated)

    cleanup_distributed()


if __name__ == "__main__":
    main()