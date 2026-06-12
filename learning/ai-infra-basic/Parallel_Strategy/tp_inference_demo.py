# tp_attention_inference_demo.py

"""
教学目标：
    用完整代码理解大模型推理中的 Tensor Parallelism，张量并行。

本代码实现的是一个带 Attention 的 TP 版 Decoder-only Transformer，
结构类似 GPT / LLaMA / Qwen 的简化版本。

整体结构：

    token embedding
        ↓
    position embedding
        ↓
    TPTransformerBlock × N
        ├── TP Causal Self-Attention
        │       ├── q_proj: Column Parallel
        │       ├── k_proj: Column Parallel
        │       ├── v_proj: Column Parallel
        │       ├── local heads attention
        │       └── o_proj: Row Parallel + all_reduce
        │
        └── TP FeedForward
                ├── up_proj: Column Parallel
                ├── activation
                └── down_proj: Row Parallel + all_reduce
        ↓
    final norm
        ↓
    lm_head
        ↓
    logits

重点理解：
    1. TP 不是切 batch，而是切模型内部矩阵；
    2. ColumnParallelLinear 按输出维度切；
    3. RowParallelLinear 按输入维度切；
    4. Attention 中 q/k/v 通常按 head 切，也就是 Column Parallel；
    5. Attention 的 o_proj 用 Row Parallel，把不同 rank 的 head 输出合并；
    6. FFN 中 up_proj 用 Column Parallel，down_proj 用 Row Parallel；
    7. Row Parallel 的合并方式是 all_reduce_sum，不是 all_gather；
    8. 一个请求会同时进入所有 TP rank 共同计算。
"""

import os
import math
import argparse
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ============================================================
# 1. 分布式初始化
# ============================================================

def init_distributed(use_cpu: bool = False):
    """
    初始化 torch.distributed。

    运行方式通常是：

        torchrun --nproc_per_node=2 tp_attention_inference_demo.py

    torchrun 会为每个进程设置环境变量：

        RANK:
            当前进程在全局通信世界中的编号。

        WORLD_SIZE:
            总进程数。

        LOCAL_RANK:
            当前进程在本机上的编号。

    在本教学代码里：
        WORLD_SIZE 就等价于 TP size。

    例如：

        torchrun --nproc_per_node=2 ...

        会启动两个进程：

            rank 0 / local_rank 0
            rank 1 / local_rank 1

        world_size = 2

    在 TP 中：
        world_size 表示矩阵被切成几份；
        rank 表示当前进程负责第几份。
    """

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if use_cpu:
        # CPU 模式用 gloo 后端。
        # 方便没有 GPU 时理解代码逻辑。
        backend = "gloo"
        device = torch.device("cpu")
    else:
        # GPU 模式通常用 nccl 后端。
        # NCCL 是 NVIDIA GPU 上常用的高性能通信库。
        backend = "nccl"

        # 每个进程绑定一张 GPU。
        #
        # rank 和 local_rank 不是同一个概念。
        # 在单机多卡时它们经常相同；
        # 在多机多卡时，rank 是全局编号，local_rank 是本机编号。
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    return rank, world_size, local_rank, device


def cleanup_distributed():
    """
    销毁分布式进程组。
    """
    dist.destroy_process_group()


# ============================================================
# 2. 通信辅助函数
# ============================================================

def all_reduce_sum(x: torch.Tensor) -> torch.Tensor:
    """
    对所有 TP rank 上的 tensor 做求和。

    输入：
        每个 rank 都有一个 partial tensor。

    输出：
        每个 rank 都得到相同的 sum 结果。

    在 RowParallelLinear 中使用：

        Y = X0 @ W0 + X1 @ W1 + ...

    每个 rank 先计算自己的 partial_y，
    然后通过 all_reduce_sum 把所有 partial_y 相加。
    """

    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x


def all_gather_last_dim(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    沿最后一个维度执行 all_gather。

    用于 ColumnParallelLinear 在需要完整输出时使用。

    举例：
        rank 0:
            x0.shape = [B, S, O/2]

        rank 1:
            x1.shape = [B, S, O/2]

    all_gather 后，每个 rank 都有：

        [x0, x1]

    拼接后：

        y.shape = [B, S, O]

    注意：
        高性能 TP 中并不是每个 ColumnParallelLinear 后都 all_gather。
        很多时候会保持输出分片状态，直接交给后面的 RowParallelLinear。
    """

    gather_list = [
        torch.empty_like(x)
        for _ in range(world_size)
    ]

    dist.all_gather(gather_list, x)

    return torch.cat(gather_list, dim=-1)


# ============================================================
# 3. ColumnParallelLinear
# ============================================================

class ColumnParallelLinear(nn.Module):
    """
    Column Parallel Linear，按输出维度切分 Linear 权重。

    普通 Linear：

        y = x @ W + b

    如果使用 PyTorch nn.Linear：
        weight.shape = [out_features, in_features]

    Column Parallel 的思想：

        完整 W 按 out_features 切成多份：

            W = [W0; W1; W2; ...]

        注意这里按 PyTorch weight 形状看，是按第 0 维切；
        按数学矩阵 W^T 看，可以理解成按输出列切。

    每个 rank 只保存一部分输出维度对应的权重：

        rank i:
            weight_i.shape = [out_features / tp_size, in_features]

    每个 rank 计算：

        local_y_i = x @ W_i^T

    local_y_i.shape:

        [B, S, out_features / tp_size]

    如果后续需要完整输出，就 all_gather；
    如果后续可以继续使用分片输出，就不 gather。

    在 Transformer 中常见用途：
        1. Attention 的 q_proj / k_proj / v_proj；
        2. FFN 的 up_proj / gate_proj。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        world_size: int,
        bias: bool = True,
        gather_output: bool = False,
    ):
        super().__init__()

        assert out_features % world_size == 0, (
            "ColumnParallelLinear 要求 out_features 能被 world_size 整除"
        )

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.gather_output = gather_output

        # 当前 rank 负责的输出维度数量。
        self.local_out_features = out_features // world_size

        # 当前 rank 只保存完整 weight 的一个输出切片。
        #
        # 完整 weight:
        #   [out_features, in_features]
        #
        # 当前 rank:
        #   [local_out_features, in_features]
        self.weight = nn.Parameter(
            torch.empty(self.local_out_features, in_features)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.local_out_features)
            )
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化本 rank 的权重切片。

        教学代码中每个 rank 独立初始化。
        真实系统中通常会从 checkpoint 中加载对应 shard。
        """

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x.shape = [B, S, in_features]

        当前 rank 输出：
            local_y.shape = [B, S, out_features / tp_size]

        如果 gather_output=True：
            返回完整 y:
                [B, S, out_features]

        如果 gather_output=False：
            返回局部 y:
                [B, S, out_features / tp_size]
        """

        # F.linear(x, weight, bias) 等价于:
        #
        #   x @ weight.T + bias
        #
        # local_y.shape:
        #   [B, S, local_out_features]
        local_y = F.linear(x, self.weight, self.bias)

        if self.gather_output:
            # 所有 rank 收集彼此的输出切片，并沿最后一维拼接。
            return all_gather_last_dim(local_y, self.world_size)

        # 不 gather，保持分片状态。
        return local_y


# ============================================================
# 4. RowParallelLinear
# ============================================================

class RowParallelLinear(nn.Module):
    """
    Row Parallel Linear，按输入维度切分 Linear 权重。

    普通 Linear：

        y = x @ W + b

    PyTorch weight.shape:
        [out_features, in_features]

    Row Parallel 的思想：

        按 in_features 切分输入 x 和权重 W。

        x = [x0 | x1 | x2 | ...]

        W =
        [
            W0
            W1
            W2
            ...
        ]

    每个 rank 保存：

        weight_i.shape = [out_features, in_features / tp_size]

    每个 rank 输入：

        x_i.shape = [B, S, in_features / tp_size]

    每个 rank 计算：

        partial_y_i = x_i @ W_i^T

    注意：

        partial_y_i.shape = [B, S, out_features]

    它已经是完整输出维度，只是数值上还不完整。

    完整输出是所有 partial_y 的求和：

        y = partial_y_0 + partial_y_1 + ...

    所以 RowParallelLinear 使用 all_reduce_sum，
    而不是 all_gather。

    在 Transformer 中常见用途：
        1. Attention 的 o_proj；
        2. FFN 的 down_proj。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        world_size: int,
        bias: bool = True,
        input_is_parallel: bool = True,
    ):
        super().__init__()

        assert in_features % world_size == 0, (
            "RowParallelLinear 要求 in_features 能被 world_size 整除"
        )

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.world_size = world_size
        self.input_is_parallel = input_is_parallel

        # 当前 rank 负责的输入维度数量。
        self.local_in_features = in_features // world_size

        # 当前 rank 只保存完整 weight 的一部分输入维度。
        #
        # 完整 weight:
        #   [out_features, in_features]
        #
        # 当前 rank:
        #   [out_features, local_in_features]
        self.weight = nn.Parameter(
            torch.empty(out_features, self.local_in_features)
        )

        # bias 是输出维度上的参数。
        #
        # 教学实现里每个 rank 都保存完整 bias，
        # all_reduce 后每个 rank 都加同样的 bias。
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.bias = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def split_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        如果输入 x 还是完整的 hidden，则按最后一维切出当前 rank 的部分。

        x.shape:
            [B, S, in_features]

        切分后：
            [B, S, in_features / tp_size]

        在标准 TP Transformer 中，
        RowParallelLinear 的输入通常已经是上一个 ColumnParallelLinear 的局部输出，
        所以一般 input_is_parallel=True，不需要再切。
        """

        chunks = torch.chunk(x, self.world_size, dim=-1)
        return chunks[self.rank].contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            如果 input_is_parallel=True:
                x.shape = [B, S, in_features / tp_size]

            如果 input_is_parallel=False:
                x.shape = [B, S, in_features]
                内部会按最后一维切分。

        输出：
            y.shape = [B, S, out_features]

        每个 rank 最终都会得到完整输出。
        """

        if self.input_is_parallel:
            local_x = x
        else:
            local_x = self.split_input(x)

        # 当前 rank 计算局部贡献。
        #
        # local_x.shape:
        #   [B, S, local_in_features]
        #
        # weight.shape:
        #   [out_features, local_in_features]
        #
        # partial_y.shape:
        #   [B, S, out_features]
        partial_y = F.linear(local_x, self.weight, bias=None)

        # 所有 rank 的 partial_y 相加，得到完整 y。
        #
        # 注意：
        #   这里是 all_reduce_sum，不是 all_gather。
        #
        # 因为每个 partial_y 都已经是完整输出维度，
        # 只是缺少其他输入切片带来的加法贡献。
        y = all_reduce_sum(partial_y)

        if self.bias is not None:
            y = y + self.bias

        return y


# ============================================================
# 5. TP Causal Self-Attention
# ============================================================

class TPCausalSelfAttention(nn.Module):
    """
    Tensor Parallel 版本的 Causal Self-Attention。

    普通 self-attention：

        x
        ↓
        q = q_proj(x)
        k = k_proj(x)
        v = v_proj(x)
        ↓
        reshape 成多个 heads
        ↓
        attention(q, k, v)
        ↓
        concat heads
        ↓
        o_proj
        ↓
        out

    TP 版本切法：

        q_proj / k_proj / v_proj:
            Column Parallel

            hidden_size -> hidden_size

            每个 rank 只输出 hidden_size / tp_size，
            也就是只负责一部分 attention heads。

        attention:
            每个 rank 只在自己的 local heads 上计算。
            不需要跨 rank 通信。

        o_proj:
            Row Parallel

            每个 rank 有 local heads 的输出，
            经过 o_proj 的权重切片得到 partial_out，
            然后 all_reduce_sum 得到完整 hidden。

    假设：
        hidden_size = 128
        num_heads = 8
        tp_size = 2

    那么：
        head_dim = 16
        每个 rank 负责 local_num_heads = 4
        local_hidden_size = 4 * 16 = 64

    rank 0:
        负责 head 0~3

    rank 1:
        负责 head 4~7
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rank: int,
        world_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert hidden_size % num_heads == 0, (
            "hidden_size 必须能被 num_heads 整除"
        )

        assert num_heads % world_size == 0, (
            "num_heads 必须能被 TP world_size 整除，"
            "因为每个 TP rank 负责一部分 attention heads"
        )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.rank = rank
        self.world_size = world_size

        self.head_dim = hidden_size // num_heads

        # 每个 rank 负责的 attention head 数量。
        self.local_num_heads = num_heads // world_size

        # 当前 rank 的 q/k/v 输出维度。
        self.local_hidden_size = self.local_num_heads * self.head_dim

        self.dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------
        # q/k/v projection: Column Parallel
        # ------------------------------------------------------------
        #
        # 完整 q_proj:
        #   hidden_size -> hidden_size
        #
        # 当前 rank:
        #   hidden_size -> hidden_size / tp_size
        #
        # 这等价于当前 rank 只生成一部分 heads 的 q。
        self.q_proj = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            rank=rank,
            world_size=world_size,
            bias=True,
            gather_output=False,
        )

        self.k_proj = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            rank=rank,
            world_size=world_size,
            bias=True,
            gather_output=False,
        )

        self.v_proj = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            rank=rank,
            world_size=world_size,
            bias=True,
            gather_output=False,
        )

        # ------------------------------------------------------------
        # output projection: Row Parallel
        # ------------------------------------------------------------
        #
        # 完整 o_proj:
        #   hidden_size -> hidden_size
        #
        # 当前 rank 只有 local_hidden_size 的 attention 输出。
        #
        # RowParallelLinear 会让每个 rank 计算：
        #   partial_out_i.shape = [B, S, hidden_size]
        #
        # 然后 all_reduce_sum 得到完整 out。
        self.o_proj = RowParallelLinear(
            in_features=hidden_size,
            out_features=hidden_size,
            rank=rank,
            world_size=world_size,
            bias=True,
            input_is_parallel=True,
        )

    def _shape_qkv(self, x: torch.Tensor) -> torch.Tensor:
        """
        把 ColumnParallelLinear 的局部输出 reshape 成 local heads。

        输入：
            x.shape = [B, S, local_hidden_size]

        其中：
            local_hidden_size = local_num_heads * head_dim

        输出：
            x.shape = [B, local_num_heads, S, head_dim]
        """

        B, S, _ = x.shape

        x = x.view(B, S, self.local_num_heads, self.head_dim)

        x = x.transpose(1, 2).contiguous()

        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        把 local heads 合并回 local hidden。

        输入：
            x.shape = [B, local_num_heads, S, head_dim]

        输出：
            x.shape = [B, S, local_hidden_size]
        """

        B, local_heads, S, D = x.shape

        x = x.transpose(1, 2).contiguous()

        x = x.view(B, S, local_heads * D)

        return x

    def _build_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        构造因果 mask，防止 token 看到未来 token。

        对 decoder-only LM 来说，训练和 prefill 阶段都需要 causal mask。

        seq_len = 5 时：

            1 0 0 0 0
            1 1 0 0 0
            1 1 1 0 0
            1 1 1 1 0
            1 1 1 1 1

        返回 shape：
            [1, 1, seq_len, seq_len]

        可以广播到 attention scores：

            scores.shape = [B, local_num_heads, S, S]
        """

        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=device,
            )
        )

        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x.shape = [B, S, hidden_size]

        输出：
            out.shape = [B, S, hidden_size]

        每个 TP rank 都会得到完整 hidden_size 输出。
        """

        B, S, H = x.shape
        device = x.device

        # ------------------------------------------------------------
        # 1. q/k/v Column Parallel projection
        # ------------------------------------------------------------
        #
        # 每个 rank 只得到 q/k/v 的一个切片。
        #
        # q_local.shape:
        #   [B, S, hidden_size / tp_size]
        q_local = self.q_proj(x)
        k_local = self.k_proj(x)
        v_local = self.v_proj(x)

        # ------------------------------------------------------------
        # 2. reshape 成 local heads
        # ------------------------------------------------------------
        #
        # q.shape:
        #   [B, local_num_heads, S, head_dim]
        q = self._shape_qkv(q_local)
        k = self._shape_qkv(k_local)
        v = self._shape_qkv(v_local)

        # ------------------------------------------------------------
        # 3. 在本 rank 的 local heads 上计算 attention
        # ------------------------------------------------------------
        #
        # scores.shape:
        #   [B, local_num_heads, S, S]
        #
        # 注意：
        #   这里不需要跨 rank 通信。
        #   因为不同 attention heads 之间是独立计算的。
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        causal_mask = self._build_causal_mask(
            seq_len=S,
            device=device,
        )

        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # context.shape:
        #   [B, local_num_heads, S, head_dim]
        context = torch.matmul(attn_weights, v)

        # ------------------------------------------------------------
        # 4. 合并本 rank 的 local heads
        # ------------------------------------------------------------
        #
        # local_context.shape:
        #   [B, S, hidden_size / tp_size]
        local_context = self._merge_heads(context)

        # ------------------------------------------------------------
        # 5. o_proj Row Parallel
        # ------------------------------------------------------------
        #
        # 当前 rank 只有部分 heads 的输出。
        #
        # o_proj 会：
        #   1. 用当前 rank 的 local_context 计算 partial_out；
        #   2. 对所有 rank 的 partial_out 执行 all_reduce_sum；
        #   3. 返回完整 hidden_size 输出。
        out = self.o_proj(local_context)

        return out


# ============================================================
# 6. TP FeedForward
# ============================================================

class TPFeedForward(nn.Module):
    """
    Tensor Parallel 版本 FFN。

    普通 FFN：

        x
        ↓
        up_proj: hidden_size -> intermediate_size
        ↓
        GELU
        ↓
        down_proj: intermediate_size -> hidden_size

    TP 切法：

        up_proj:
            Column Parallel

            每个 rank 只计算 intermediate_size / tp_size。

        activation:
            本地执行，不需要通信。

        down_proj:
            Row Parallel

            每个 rank 基于自己的 intermediate shard 计算 partial hidden，
            然后 all_reduce_sum 得到完整 hidden。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        rank: int,
        world_size: int,
    ):
        super().__init__()

        assert intermediate_size % world_size == 0, (
            "intermediate_size 必须能被 TP world_size 整除"
        )

        # hidden_size -> intermediate_size
        #
        # Column Parallel 后，每个 rank 得到：
        #   [B, S, intermediate_size / tp_size]
        self.up_proj = ColumnParallelLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            rank=rank,
            world_size=world_size,
            bias=True,
            gather_output=False,
        )

        # intermediate_size -> hidden_size
        #
        # 输入已经是 parallel 的 intermediate shard。
        #
        # Row Parallel 会 all_reduce 得到完整 hidden。
        self.down_proj = RowParallelLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            rank=rank,
            world_size=world_size,
            bias=True,
            input_is_parallel=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x.shape = [B, S, hidden_size]

        输出：
            out.shape = [B, S, hidden_size]
        """

        # up_proj:
        #   每个 rank 只得到中间维度的一部分。
        local_intermediate = self.up_proj(x)

        # GELU 是逐元素操作。
        #
        # 因为每个 rank 已经拥有自己那部分 intermediate，
        # 所以 activation 不需要通信。
        local_intermediate = F.gelu(local_intermediate)

        # down_proj:
        #   每个 rank 计算 partial output；
        #   all_reduce_sum 合并。
        out = self.down_proj(local_intermediate)

        return out


# ============================================================
# 7. TP Transformer Block
# ============================================================

class TPTransformerBlock(nn.Module):
    """
    一个完整的 TP Transformer Block。

    使用 Pre-LN 结构：

        x
        ↓
        norm1
        ↓
        TP Causal Self-Attention
        ↓
        residual add
        ↓
        norm2
        ↓
        TP FeedForward
        ↓
        residual add

    其中：

        Attention:
            q/k/v Column Parallel
            local heads attention
            o_proj Row Parallel + all_reduce

        FFN:
            up_proj Column Parallel
            activation local
            down_proj Row Parallel + all_reduce
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        rank: int,
        world_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)

        self.attn = TPCausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            rank=rank,
            world_size=world_size,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = TPFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            rank=rank,
            world_size=world_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x.shape = [B, S, hidden_size]

        输出：
            x.shape = [B, S, hidden_size]
        """

        # Attention 子层。
        #
        # norm1(x) 是完整 hidden。
        # TP Attention 内部会切 q/k/v heads。
        attn_out = self.attn(self.norm1(x))

        # 残差连接。
        x = x + attn_out

        # FFN 子层。
        ffn_out = self.ffn(self.norm2(x))

        # 残差连接。
        x = x + ffn_out

        return x


# ============================================================
# 8. 完整 TP Causal LM
# ============================================================

class TinyTPTransformerLM(nn.Module):
    """
    完整教学版 TP Decoder-only LM。

    结构：

        input_ids
            ↓
        token_embedding
            ↓
        position_embedding
            ↓
        TPTransformerBlock × num_layers
            ↓
        final_norm
            ↓
        lm_head
            ↓
        logits

    注意：
        为了教学简单，token_embedding 和 lm_head 在每个 rank 上都保留完整一份。

    真实大模型中：
        1. embedding 也可以做 vocab parallel；
        2. lm_head 通常也可以按 vocab 维度切分；
        3. 推理时 logits 可能需要 all_gather 或 distributed sampling。
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

        assert hidden_size % world_size == 0
        assert num_heads % world_size == 0
        assert intermediate_size % world_size == 0

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.rank = rank
        self.world_size = world_size

        # 每个 rank 暂时保存完整 embedding。
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)

        # 位置编码也每个 rank 保留完整。
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.blocks = nn.ModuleList([
            TPTransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                rank=rank,
                world_size=world_size,
                dropout=0.0,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)

        # 教学版本中 lm_head 不切分。
        #
        # 真实大模型 TP 中，lm_head 可以做 vocab parallel：
        #   每个 rank 只计算一部分 vocab logits。
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids.shape = [B, S]

        返回：
            logits.shape = [B, S, vocab_size]

        在 TP 中：
            每个 rank 都会执行 forward。
            每个 rank 内部只保存和计算部分矩阵 shard。
            通信操作让所有 rank 在必要位置同步完整 hidden。
        """

        B, S = input_ids.shape
        device = input_ids.device

        # 防止超过 position embedding 上限。
        if S > self.max_seq_len:
            raise ValueError(
                f"Sequence length {S} exceeds max_seq_len {self.max_seq_len}"
            )

        position_ids = torch.arange(
            0,
            S,
            dtype=torch.long,
            device=device,
        ).unsqueeze(0).expand(B, S)

        x = self.token_embedding(input_ids)
        x = x + self.position_embedding(position_ids)

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


# ============================================================
# 9. Greedy Generate
# ============================================================

@torch.no_grad()
def greedy_generate(
    model: TinyTPTransformerLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    rank: int,
) -> torch.Tensor:
    """
    简单 greedy decoding。

    注意：
        在 TP 中，同一个请求会同时进入所有 rank。

    每个 rank 都执行相同的 generation loop：

        1. 输入当前 generated；
        2. 模型前向；
        3. 取最后一个 token 的 logits；
        4. argmax 得到 next_token；
        5. 拼接到 generated 后面。

    因为 TP block 中的 all_reduce 会保证各 rank 得到一致 hidden，
    所以各 rank 得到的 logits 理论上也是一致的。

    教学版本没有实现 KV Cache。
    实际大模型推理中会结合 KV Cache，
    每一步只输入新 token，复用历史 K/V。
    """

    model.eval()

    generated = prompt_ids.clone()

    for step in range(max_new_tokens):
        logits = model(generated)

        # 只取最后一个位置的 logits，用于预测下一个 token。
        next_token_logits = logits[:, -1, :]

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat([generated, next_token], dim=1)

        if rank == 0:
            print(
                f"[rank {rank}] step={step}, "
                f"next_token={next_token.squeeze(-1).tolist()}",
                flush=True,
            )

        if generated.size(1) >= model.max_seq_len:
            break

    return generated


# ============================================================
# 10. Debug 打印参数 shard 信息
# ============================================================

def print_tp_layout(
    rank: int,
    world_size: int,
    hidden_size: int,
    num_heads: int,
    intermediate_size: int,
):
    """
    打印当前 TP 配置，帮助理解每个 rank 负责多少东西。
    """

    head_dim = hidden_size // num_heads
    local_num_heads = num_heads // world_size
    local_hidden = hidden_size // world_size
    local_intermediate = intermediate_size // world_size

    print(
        f"[rank {rank}] TP layout:\n"
        f"  world_size             = {world_size}\n"
        f"  hidden_size            = {hidden_size}\n"
        f"  num_heads              = {num_heads}\n"
        f"  head_dim               = {head_dim}\n"
        f"  local_num_heads        = {local_num_heads}\n"
        f"  local_attention_hidden = {local_num_heads * head_dim}\n"
        f"  local_hidden_shard     = {local_hidden}\n"
        f"  intermediate_size      = {intermediate_size}\n"
        f"  local_intermediate     = {local_intermediate}\n",
        flush=True,
    )


# ============================================================
# 11. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="使用 CPU + gloo 模拟 TP",
    )
    args = parser.parse_args()

    rank, world_size, local_rank, device = init_distributed(use_cpu=args.cpu)

    # ------------------------------------------------------------
    # 模型配置
    # ------------------------------------------------------------
    #
    # 这里用一个很小的模型，方便你跑通和理解。
    #
    # 真实大模型可能是：
    #   hidden_size = 4096 / 8192
    #   num_heads = 32 / 64
    #   num_layers = 32 / 80
    #   intermediate_size = 11008 / 28672
    vocab_size = 100
    hidden_size = 128
    intermediate_size = 512
    num_heads = 8
    num_layers = 2
    max_seq_len = 64

    assert hidden_size % world_size == 0, (
        "hidden_size 必须能被 TP size 整除"
    )
    assert num_heads % world_size == 0, (
        "num_heads 必须能被 TP size 整除"
    )
    assert intermediate_size % world_size == 0, (
        "intermediate_size 必须能被 TP size 整除"
    )

    # 为了让各 rank 上未切分参数初始化一致，
    # 使用相同随机种子。
    #
    # 注意：
    #   这只是教学简化。
    #   真实系统会从同一个 checkpoint 加载参数 shard。
    torch.manual_seed(1234)

    print_tp_layout(
        rank=rank,
        world_size=world_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
    )

    model = TinyTPTransformerLM(
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

    # ------------------------------------------------------------
    # 构造输入
    # ------------------------------------------------------------
    #
    # TP 和 DP 不同：
    #
    # DP:
    #   不同请求可以分发到不同 worker。
    #
    # TP:
    #   同一个请求必须进入所有 TP rank。
    #
    # 因此所有 rank 都构造相同 input_ids。
    prompt_ids = torch.tensor(
        [[10, 20, 30, 40]],
        dtype=torch.long,
        device=device,
    )

    if rank == 0:
        print(f"prompt_ids = {prompt_ids.tolist()}", flush=True)

    # ------------------------------------------------------------
    # 执行生成
    # ------------------------------------------------------------

    generated = greedy_generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=5,
        rank=rank,
    )

    if rank == 0:
        print("\n========== Final Generated ==========")
        print(generated.tolist())

    cleanup_distributed()


if __name__ == "__main__":
    main()