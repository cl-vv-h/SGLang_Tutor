# flash_decoding_tutorial.py

"""
============================================================
FlashDecoding 教学版源码
============================================================

本代码目标：
    用 PyTorch 实现 FlashDecoding 的核心数学逻辑。

核心思想：

    Decode 阶段：
        Q 很短，通常 Q_len = 1。
        K/V 很长，是历史 KV Cache。

    普通 attention：
        scores = Q @ K.T
        probs = softmax(scores)
        out = probs @ V

    FlashDecoding：
        将 K/V 沿 sequence 维切成多个 block。
        每个 block 独立计算 partial softmax 统计量：

            m_b = max(scores_b)
            l_b = sum(exp(scores_b - m_b))
            o_b = exp(scores_b - m_b) @ V_b

        然后全局合并：

            m = max_b m_b
            scale_b = exp(m_b - m)

            l = sum_b scale_b * l_b
            o = sum_b scale_b * o_b

            out = o / l

    这样得到的 out 与完整 softmax attention 数学等价。

注意：
    这份代码是教学版。
    它不会比 PyTorch naive attention 更快。
    真正加速来自 CUDA/Triton kernel 并行加载 K/V blocks。
"""

from dataclasses import dataclass
from typing import List, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 基础配置
# ============================================================

@dataclass
class FlashDecodingConfig:
    """
    FlashDecoding 教学配置。

    num_heads:
        attention head 数。

    head_dim:
        每个 head 的维度。

    block_size:
        将 KV Cache 沿 context_len 切分的 block 大小。

        例如：
            context_len = 1024
            block_size = 128

        则共有：
            num_blocks = ceil(1024 / 128) = 8
    """

    num_heads: int = 4
    head_dim: int = 32
    block_size: int = 128


# ============================================================
# 2. Naive Decode Attention
# ============================================================

def naive_decode_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
) -> torch.Tensor:
    """
    普通 decode attention。

    输入：
        q.shape = [num_heads, 1, head_dim]

        k_cache.shape = [num_heads, context_len, head_dim]

        v_cache.shape = [num_heads, context_len, head_dim]

    输出：
        out.shape = [num_heads, 1, head_dim]

    说明：
        这是最直接的 attention 公式：

            scores = q @ k^T / sqrt(d)
            probs = softmax(scores)
            out = probs @ v

    维度：

        q:
            [H, 1, D]

        k_cache.transpose(-2, -1):
            [H, D, S]

        scores:
            [H, 1, S]

        probs:
            [H, 1, S]

        out:
            [H, 1, D]
    """

    head_dim = q.size(-1)

    scores = torch.matmul(
        q,
        k_cache.transpose(-2, -1),
    ) / math.sqrt(head_dim)

    probs = F.softmax(scores, dim=-1)

    out = torch.matmul(probs, v_cache)

    return out


# ============================================================
# 3. FlashDecoding Attention
# ============================================================

def flash_decoding_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """
    FlashDecoding 教学版实现。

    输入：
        q.shape = [num_heads, 1, head_dim]

        k_cache.shape = [num_heads, context_len, head_dim]

        v_cache.shape = [num_heads, context_len, head_dim]

    输出：
        out.shape = [num_heads, 1, head_dim]

    核心步骤：

        1. 将 K/V 沿 context_len 维切成多个 block。

        2. 每个 block 独立计算：
            scores_b = q @ k_b.T / sqrt(d)

            m_b = max(scores_b)

            exp_scores_b = exp(scores_b - m_b)

            l_b = sum(exp_scores_b)

            o_b = exp_scores_b @ v_b

        3. 对所有 block 做全局合并：

            m = max_b m_b

            scale_b = exp(m_b - m)

            global_l = sum_b scale_b * l_b

            global_o = sum_b scale_b * o_b

            out = global_o / global_l

    注意：
        这里使用 Python for-loop 模拟多个 CUDA blocks。
        真实 FlashDecoding 中，每个 block 通常由不同 CUDA block 并行处理。
    """

    num_heads, q_len, head_dim = q.shape
    assert q_len == 1, "FlashDecoding is mainly for decode where q_len=1."

    _, context_len, _ = k_cache.shape

    scale = 1.0 / math.sqrt(head_dim)

    # ------------------------------------------------------------
    # partial_m_list:
    #   每个 block 的局部最大值 m_b。
    #
    # 每个 m_b.shape = [num_heads, 1, 1]
    #
    # 为什么保留维度？
    #   方便后续 broadcast 到 [num_heads, 1, head_dim]
    # ------------------------------------------------------------
    partial_m_list: List[torch.Tensor] = []

    # partial_l_list:
    #   每个 block 的局部分母 l_b。
    #
    # 每个 l_b.shape = [num_heads, 1, 1]
    partial_l_list: List[torch.Tensor] = []

    # partial_o_list:
    #   每个 block 的未归一化局部输出 o_b。
    #
    # 每个 o_b.shape = [num_heads, 1, head_dim]
    partial_o_list: List[torch.Tensor] = []

    # ------------------------------------------------------------
    # Step 1: 遍历 KV blocks
    # ------------------------------------------------------------
    for block_start in range(0, context_len, block_size):
        block_end = min(block_start + block_size, context_len)

        # k_block.shape = [num_heads, block_len, head_dim]
        # v_block.shape = [num_heads, block_len, head_dim]
        k_block = k_cache[:, block_start:block_end, :]
        v_block = v_cache[:, block_start:block_end, :]

        # block_len 可能小于 block_size，比如最后一个 block。
        block_len = k_block.size(1)

        # scores_b:
        #
        # q.shape = [H, 1, D]
        # k_block.transpose(-2,-1).shape = [H, D, block_len]
        #
        # scores_b.shape = [H, 1, block_len]
        scores_b = torch.matmul(
            q,
            k_block.transpose(-2, -1),
        ) * scale

        # m_b 是当前 block 内的最大 score。
        #
        # m_b.shape = [H, 1, 1]
        m_b = scores_b.max(dim=-1, keepdim=True).values

        # exp_scores_b.shape = [H, 1, block_len]
        exp_scores_b = torch.exp(scores_b - m_b)

        # l_b 是当前 block 的局部分母。
        #
        # l_b.shape = [H, 1, 1]
        l_b = exp_scores_b.sum(dim=-1, keepdim=True)

        # o_b 是当前 block 的未归一化局部输出。
        #
        # exp_scores_b.shape = [H, 1, block_len]
        # v_block.shape = [H, block_len, D]
        #
        # o_b.shape = [H, 1, D]
        o_b = torch.matmul(exp_scores_b, v_block)

        partial_m_list.append(m_b)
        partial_l_list.append(l_b)
        partial_o_list.append(o_b)

    # ------------------------------------------------------------
    # Step 2: 合并所有 partial results
    # ------------------------------------------------------------

    # partial_m.shape = [num_blocks, H, 1, 1]
    partial_m = torch.stack(partial_m_list, dim=0)

    # partial_l.shape = [num_blocks, H, 1, 1]
    partial_l = torch.stack(partial_l_list, dim=0)

    # partial_o.shape = [num_blocks, H, 1, D]
    partial_o = torch.stack(partial_o_list, dim=0)

    # 全局最大值：
    #
    # global_m.shape = [H, 1, 1]
    global_m = partial_m.max(dim=0).values

    # rescale:
    #
    # 每个 block 的 softmax 都是用自己的 m_b 做稳定化。
    # 要合并成全局 softmax，需要把它们转到同一个 global_m 标尺。
    #
    # rescale_b = exp(m_b - global_m)
    #
    # rescale.shape = [num_blocks, H, 1, 1]
    rescale = torch.exp(partial_m - global_m.unsqueeze(0))

    # 全局分母：
    #
    # global_l = sum_b exp(m_b - global_m) * l_b
    #
    # global_l.shape = [H, 1, 1]
    global_l = (rescale * partial_l).sum(dim=0)

    # 全局分子：
    #
    # global_o = sum_b exp(m_b - global_m) * o_b
    #
    # rescale.shape = [num_blocks, H, 1, 1]
    # partial_o.shape = [num_blocks, H, 1, D]
    #
    # global_o.shape = [H, 1, D]
    global_o = (rescale * partial_o).sum(dim=0)

    # 最终输出：
    #
    # out.shape = [H, 1, D]
    out = global_o / global_l

    return out


# ============================================================
# 4. 验证 FlashDecoding 与 Naive Attention 一致
# ============================================================

def verify_flash_decoding_correctness():
    """
    验证教学版 FlashDecoding 和 naive attention 输出一致。

    因为两者数学等价，所以误差应该非常小。
    """

    torch.manual_seed(0)

    num_heads = 4
    head_dim = 32
    context_len = 1024
    block_size = 128

    # q.shape = [H, 1, D]
    q = torch.randn(num_heads, 1, head_dim)

    # k/v.shape = [H, S, D]
    k = torch.randn(num_heads, context_len, head_dim)
    v = torch.randn(num_heads, context_len, head_dim)

    naive_out = naive_decode_attention(q, k, v)

    flash_out = flash_decoding_attention(
        q=q,
        k_cache=k,
        v_cache=v,
        block_size=block_size,
    )

    max_diff = (naive_out - flash_out).abs().max().item()

    print("\n=== Correctness Check ===")
    print(f"naive_out.shape = {tuple(naive_out.shape)}")
    print(f"flash_out.shape = {tuple(flash_out.shape)}")
    print(f"max_diff = {max_diff:.8f}")


# ============================================================
# 5. 支持 FlashDecoding 的 Tiny Attention
# ============================================================

class TinyDecodeAttention(nn.Module):
    """
    一个只演示 decode 阶段的 Attention 层。

    它包含：
        q_proj
        k_proj
        v_proj
        o_proj

    在 decode 时：
        1. 输入当前 token hidden x_new；
        2. 计算 q_new, k_new, v_new；
        3. 将 k_new/v_new append 到 KV Cache；
        4. 使用 FlashDecoding 从完整 KV Cache 计算 attention output；
        5. 经过 o_proj 得到最终输出。

    注意：
        这里为了教学简单，KV Cache 是直接拼接 tensor。
        工业实现中 KV Cache 通常是 block/page 管理。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        block_size: int,
    ):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.block_size = block_size

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x.shape = [1, 1, hidden_size]

        输出：
            x.shape = [num_heads, 1, head_dim]
        """

        B, T, H = x.shape
        assert B == 1
        assert T == 1

        x = x.view(B, T, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x.squeeze(0)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x.shape = [num_heads, 1, head_dim]

        输出：
            x.shape = [1, 1, hidden_size]
        """

        Hn, T, Dh = x.shape
        x = x.permute(1, 0, 2).contiguous()
        x = x.view(1, T, Hn * Dh)
        return x

    def forward_decode(
        self,
        x_new: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行一次 decode attention。

        输入：
            x_new.shape = [1, 1, hidden_size]

            k_cache.shape = [num_heads, past_len, head_dim]

            v_cache.shape = [num_heads, past_len, head_dim]

        输出：
            out.shape = [1, 1, hidden_size]

            new_k_cache.shape = [num_heads, past_len + 1, head_dim]

            new_v_cache.shape = [num_heads, past_len + 1, head_dim]

        说明：
            当前 token 的 K/V 会被追加到 cache 中。
            然后 Q_new attend 到完整 K/V。
        """

        q = self._split_heads(self.q_proj(x_new))
        k_new = self._split_heads(self.k_proj(x_new))
        v_new = self._split_heads(self.v_proj(x_new))

        # append 当前 token 的 K/V。
        #
        # k_new.shape = [H, 1, D]
        # k_cache.shape = [H, past_len, D]
        #
        # new_k_cache.shape = [H, past_len+1, D]
        new_k_cache = torch.cat([k_cache, k_new], dim=1)
        new_v_cache = torch.cat([v_cache, v_new], dim=1)

        # 使用 FlashDecoding 计算 decode attention。
        #
        # q.shape = [H, 1, D]
        # new_k_cache.shape = [H, context_len, D]
        # new_v_cache.shape = [H, context_len, D]
        #
        # context.shape = [H, 1, D]
        context = flash_decoding_attention(
            q=q,
            k_cache=new_k_cache,
            v_cache=new_v_cache,
            block_size=self.block_size,
        )

        context = self._merge_heads(context)

        out = self.o_proj(context)

        return out, new_k_cache, new_v_cache


# ============================================================
# 6. Tiny Decode Demo
# ============================================================

def demo_tiny_decode_attention():
    """
    演示一个带 FlashDecoding 的 decode attention。

    这里模拟已经有 past KV Cache：

        past_len = 1024

    然后每次 decode 一个新 token。
    """

    torch.manual_seed(0)

    hidden_size = 128
    num_heads = 4
    head_dim = hidden_size // num_heads
    block_size = 128
    past_len = 1024

    attn = TinyDecodeAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        block_size=block_size,
    )

    # 模拟历史 KV Cache。
    #
    # k_cache.shape = [H, past_len, D]
    # v_cache.shape = [H, past_len, D]
    k_cache = torch.randn(num_heads, past_len, head_dim)
    v_cache = torch.randn(num_heads, past_len, head_dim)

    # 当前新 token hidden。
    #
    # x_new.shape = [1, 1, hidden_size]
    x_new = torch.randn(1, 1, hidden_size)

    out, new_k_cache, new_v_cache = attn.forward_decode(
        x_new=x_new,
        k_cache=k_cache,
        v_cache=v_cache,
    )

    print("\n=== Tiny Decode Attention Demo ===")
    print(f"x_new.shape       = {tuple(x_new.shape)}")
    print(f"k_cache.shape     = {tuple(k_cache.shape)}")
    print(f"new_k_cache.shape = {tuple(new_k_cache.shape)}")
    print(f"out.shape         = {tuple(out.shape)}")


# ============================================================
# 7. 展示 block 级 partial 结果
# ============================================================

def debug_flash_decoding_blocks():
    """
    用很小的例子打印每个 block 的 m/l/o 维度，
    帮助理解 FlashDecoding 的中间变量。
    """

    torch.manual_seed(0)

    num_heads = 2
    head_dim = 4
    context_len = 10
    block_size = 4

    q = torch.randn(num_heads, 1, head_dim)
    k = torch.randn(num_heads, context_len, head_dim)
    v = torch.randn(num_heads, context_len, head_dim)

    print("\n=== Debug FlashDecoding Blocks ===")
    print(f"q.shape = {tuple(q.shape)}")
    print(f"k.shape = {tuple(k.shape)}")
    print(f"v.shape = {tuple(v.shape)}")

    scale = 1.0 / math.sqrt(head_dim)

    for block_start in range(0, context_len, block_size):
        block_end = min(block_start + block_size, context_len)

        k_block = k[:, block_start:block_end, :]
        v_block = v[:, block_start:block_end, :]

        scores_b = torch.matmul(q, k_block.transpose(-2, -1)) * scale
        m_b = scores_b.max(dim=-1, keepdim=True).values
        exp_scores_b = torch.exp(scores_b - m_b)
        l_b = exp_scores_b.sum(dim=-1, keepdim=True)
        o_b = torch.matmul(exp_scores_b, v_block)

        print(
            f"block [{block_start}, {block_end}): "
            f"k_block={tuple(k_block.shape)}, "
            f"scores_b={tuple(scores_b.shape)}, "
            f"m_b={tuple(m_b.shape)}, "
            f"l_b={tuple(l_b.shape)}, "
            f"o_b={tuple(o_b.shape)}"
        )


# ============================================================
# 8. Main
# ============================================================

if __name__ == "__main__":
    verify_flash_decoding_correctness()
    debug_flash_decoding_blocks()
    demo_tiny_decode_attention()