# flash_attention_tutorial.py

"""
============================================================
FlashAttention 教学版源码
============================================================

本代码目标：
    用 PyTorch 写出 FlashAttention forward 的核心逻辑。

重点：
    1. 不显式保存完整 attention matrix；
    2. Q/K/V 按 block 切分；
    3. 对每个 Q block，遍历 K/V blocks；
    4. 使用 online softmax 合并 partial attention；
    5. 支持 causal mask；
    6. 验证输出和 naive attention 一致。

注意：
    这不是高性能实现。
    真实 FlashAttention 需要 CUDA/Triton kernel，
    将 block 搬入 SRAM/shared memory 中计算。
"""

from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 配置
# ============================================================

@dataclass
class FlashAttentionConfig:
    """
    FlashAttention 教学配置。

    q_block_size:
        Q 沿 sequence 维的 block 大小。

    kv_block_size:
        K/V 沿 sequence 维的 block 大小。

    causal:
        是否使用 causal mask。
        Decoder-only LLM 通常为 True。
    """

    q_block_size: int = 16
    kv_block_size: int = 16
    causal: bool = True


# ============================================================
# 2. Naive Attention
# ============================================================

def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """
    普通 attention 实现。

    输入：
        q.shape = [B, H, S, D]
        k.shape = [B, H, S, D]
        v.shape = [B, H, S, D]

    输出：
        out.shape = [B, H, S, D]

    普通公式：
        scores = q @ k^T / sqrt(D)
        probs = softmax(scores)
        out = probs @ v

    这里会显式构造：
        scores.shape = [B, H, S, S]
        probs.shape  = [B, H, S, S]

    这正是 FlashAttention 想避免的。
    """

    B, H, S, D = q.shape

    scores = torch.matmul(
        q,
        k.transpose(-2, -1),
    ) / math.sqrt(D)

    if causal:
        mask = torch.tril(
            torch.ones(S, S, dtype=torch.bool, device=q.device)
        )

        scores = scores.masked_fill(mask.view(1, 1, S, S) == 0, float("-inf"))

    probs = F.softmax(scores, dim=-1)

    out = torch.matmul(probs, v)

    return out


# ============================================================
# 3. FlashAttention Forward 教学版
# ============================================================
def flash_attention_forward_tutorial(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_block_size: int,
    kv_block_size: int,
    causal: bool = True,
) -> torch.Tensor:
    """
    FlashAttention forward 教学版。

    输入：
        q.shape = [B, H, S, D]
        k.shape = [B, H, S, D]
        v.shape = [B, H, S, D]

    其中：
        B = batch size
            一次处理多少个样本。

        H = num_heads
            Attention head 数量。

        S = seq_len
            序列长度，也就是 token 数量。

        D = head_dim
            每个 attention head 的维度。

    输出：
        out.shape = [B, H, S, D]

    普通 Attention 的完整公式是：

        scores = q @ k.transpose(-2, -1) / sqrt(D)
        probs  = softmax(scores, dim=-1)
        out    = probs @ v

    其中：
        scores.shape = [B, H, S, S]
        probs.shape  = [B, H, S, S]

    FlashAttention 的核心是：
        不显式构造完整的 [S, S] scores/probs 矩阵，
        而是把 Q 和 K/V 都切成 block，
        对每个 Q block，逐个扫描 K/V block，
        使用 online softmax 累积最终结果。

    q_block_size:
        每个 Q block 包含多少个 query token。

    kv_block_size:
        每个 K/V block 包含多少个 key/value token。
    """

    # ------------------------------------------------------------
    # 1. 读取输入张量的维度
    # ------------------------------------------------------------
    #
    # q.shape = [B, H, S, D]
    #
    # B:
    #   batch size
    #
    # H:
    #   attention head 数
    #
    # S:
    #   sequence length
    #
    # D:
    #   每个 head 的维度
    B, H, S, D = q.shape

    # ------------------------------------------------------------
    # 2. Attention scale
    # ------------------------------------------------------------
    #
    # 标准 attention 里会做：
    #
    #   scores = QK^T / sqrt(D)
    #
    # 为什么要除以 sqrt(D)？
    #
    # 因为 Q 和 K 的维度 D 越大，点积结果的数值方差越大。
    # 如果不缩放，scores 可能很大，softmax 后容易变得极端，
    # 导致梯度不稳定。
    #
    # scale = 1 / sqrt(D)
    scale = 1.0 / math.sqrt(D)

    # ------------------------------------------------------------
    # 3. 创建输出张量
    # ------------------------------------------------------------
    #
    # out.shape = [B, H, S, D]
    #
    # 它最终保存每个 query token 的 attention 输出。
    #
    # 注意：
    #   FlashAttention 不保存完整 scores/probs，
    #   但最终输出 out 还是必须保存。
    out = torch.empty_like(q)

    # ============================================================
    # 外层循环：按 Q 的 sequence 维切 block
    # ============================================================
    #
    # 普通 attention 是一次性处理全部 Q：
    #
    #   Q.shape = [B, H, S, D]
    #
    # FlashAttention 把 Q 按 token 维切成多个小块。
    #
    # 例如：
    #   S = 64
    #   q_block_size = 16
    #
    # 则：
    #   Q block 0: token 0  ~ 15
    #   Q block 1: token 16 ~ 31
    #   Q block 2: token 32 ~ 47
    #   Q block 3: token 48 ~ 63
    #
    # 对每一个 Q block，单独计算它对所有 K/V 的 attention 输出。
    for q_start in range(0, S, q_block_size):

        # 当前 Q block 的结束位置。
        #
        # 用 min 是因为最后一个 block 可能不满 q_block_size。
        #
        # 例如：
        #   S = 70
        #   q_block_size = 16
        #
        # 最后一个 block:
        #   q_start = 64
        #   q_end = 70
        q_end = min(q_start + q_block_size, S)

        # --------------------------------------------------------
        # 4. 取出当前 Q block
        # --------------------------------------------------------
        #
        # q_block.shape = [B, H, Qb, D]
        #
        # Qb = 当前 Q block 的实际长度。
        #
        # 大多数情况下：
        #   Qb = q_block_size
        #
        # 最后一个 block 可能：
        #   Qb < q_block_size
        q_block = q[:, :, q_start:q_end, :]

        # 当前 Q block 中 query token 的数量。
        #
        # Qb 是 Query Block size 的实际值。
        #
        # 例如：
        #   q_block.shape = [2, 4, 16, 32]
        #   Qb = 16
        Qb = q_block.size(2)

        # --------------------------------------------------------
        # 5. 当前 Q block 中每个 query token 的绝对位置
        # --------------------------------------------------------
        #
        # q_positions.shape = [Qb]
        #
        # 例如：
        #   q_start = 16
        #   q_end = 32
        #
        # 则：
        #   q_positions = [16, 17, 18, ..., 31]
        #
        # 它主要用于 causal mask。
        #
        # 对 Decoder-only LLM 来说：
        #   query position i 只能看 key position <= i 的 token。
        q_positions = torch.arange(
            q_start,
            q_end,
            device=q.device,
        )

        # ========================================================
        # 初始化 online softmax 状态
        # ========================================================
        #
        # 对当前 Q block 来说，我们要让它 attend 所有 K/V token。
        #
        # 但是 FlashAttention 不一次性处理所有 K/V，
        # 而是按 K/V block 逐块扫描。
        #
        # 因此需要为当前 Q block 维护三个累计状态：
        #
        #   m:
        #       当前已经扫描过的 K/V block 中，
        #       每个 query row 的最大 score。
        #
        #   l:
        #       当前已经扫描过的 K/V block 中，
        #       每个 query row 的 softmax 分母。
        #
        #   acc:
        #       当前已经扫描过的 K/V block 中，
        #       每个 query row 的未归一化输出分子。
        #
        # 最后：
        #
        #   out_block = acc / l
        #
        # 这三个变量就是 FlashAttention 的核心。
        # ========================================================

        # --------------------------------------------------------
        # 6. m: 每一行 query 当前见过的最大 score
        # --------------------------------------------------------
        #
        # m.shape = [B, H, Qb]
        #
        # 为什么是 [B, H, Qb]？
        #
        # 因为对于每个 batch、每个 head、每个 query token，
        # 都需要单独维护一个最大值。
        #
        # 初始值为 -inf，表示还没有看过任何 key。
        #
        # 举例：
        #   当前 Q block 有 16 个 query token，
        #   那么每个 query token 都有自己的 m。
        m = torch.full(
            (B, H, Qb),
            -float("inf"),
            device=q.device,
            dtype=q.dtype,
        )

        # --------------------------------------------------------
        # 7. l: 每一行 query 当前的 softmax 分母
        # --------------------------------------------------------
        #
        # l.shape = [B, H, Qb]
        #
        # 对某个 query 来说，softmax 分母是：
        #
        #   sum_j exp(score_j - m)
        #
        # 由于我们分 block 扫描 K/V，
        # 所以 l 是逐 block 累积出来的。
        #
        # 初始为 0，表示还没有累计任何 key。
        l = torch.zeros(
            (B, H, Qb),
            device=q.device,
            dtype=q.dtype,
        )

        # --------------------------------------------------------
        # 8. acc: 每一行 query 当前的未归一化输出分子
        # --------------------------------------------------------
        #
        # acc.shape = [B, H, Qb, D]
        #
        # 普通 attention 输出：
        #
        #   out_i = Σ_j softmax(scores_i)[j] * V_j
        #
        # 也可以写成：
        #
        #   out_i =
        #       sum_j exp(score_ij - m) * V_j
        #       /
        #       sum_j exp(score_ij - m)
        #
        # 其中：
        #   分子就是 acc
        #   分母就是 l
        #
        # 所以 acc 累积的是：
        #
        #   sum_j exp(score_ij - m) * V_j
        #
        # 初始为 0。
        acc = torch.zeros(
            (B, H, Qb, D),
            device=q.device,
            dtype=q.dtype,
        )

        # ========================================================
        # 内层循环：按 K/V 的 sequence 维切 block
        # ========================================================
        #
        # 对当前 q_block，要扫描所有 K/V block。
        #
        # 例如：
        #   S = 64
        #   kv_block_size = 16
        #
        # 则：
        #   KV block 0: token 0  ~ 15
        #   KV block 1: token 16 ~ 31
        #   KV block 2: token 32 ~ 47
        #   KV block 3: token 48 ~ 63
        #
        # 对每个 KV block：
        #   1. 计算局部 scores
        #   2. 做局部 softmax 统计
        #   3. 用 online softmax 合并到 m/l/acc
        for kv_start in range(0, S, kv_block_size):

            # 当前 K/V block 的结束位置。
            kv_end = min(kv_start + kv_block_size, S)

            # ----------------------------------------------------
            # 9. causal 场景下，如果整个 KV block 都在未来，可以跳过
            # ----------------------------------------------------
            #
            # 对 causal attention：
            #
            #   query position i 只能看 key position <= i
            #
            # 如果当前 KV block 的起始位置 kv_start
            # 已经大于当前 Q block 的最大 query position q_end - 1，
            # 那么这个 KV block 对当前 Q block 完全不可见。
            #
            # 例如：
            #   Q block positions = [0,1,2,3]
            #   KV block positions = [8,9,10,11]
            #
            # 那么所有 key 都是未来 token，直接跳过。
            if causal and kv_start > q_end - 1:
                continue

            # ----------------------------------------------------
            # 10. 取出当前 K/V block
            # ----------------------------------------------------
            #
            # k_block.shape = [B, H, Kb, D]
            # v_block.shape = [B, H, Kb, D]
            #
            # Kb 是当前 KV block 的实际长度。
            #
            # 最后一个 block 可能不满 kv_block_size。
            k_block = k[:, :, kv_start:kv_end, :]
            v_block = v[:, :, kv_start:kv_end, :]

            # 当前 KV block 的实际 token 数量。
            Kb = k_block.size(2)

            # ----------------------------------------------------
            # 11. 当前 KV block 中 key token 的绝对位置
            # ----------------------------------------------------
            #
            # kv_positions.shape = [Kb]
            #
            # 例如：
            #   kv_start = 16
            #   kv_end = 32
            #
            # 则：
            #   kv_positions = [16, 17, ..., 31]
            #
            # 用于 causal mask。
            kv_positions = torch.arange(
                kv_start,
                kv_end,
                device=q.device,
            )

            # ----------------------------------------------------
            # 12. 计算当前 Q block 和当前 K block 的局部 scores
            # ----------------------------------------------------
            #
            # q_block.shape = [B, H, Qb, D]
            #
            # k_block.transpose(-2, -1).shape = [B, H, D, Kb]
            #
            # scores.shape = [B, H, Qb, Kb]
            #
            # scores[b, h, i, j] 表示：
            #   第 b 个样本，
            #   第 h 个 head，
            #   当前 Q block 中第 i 个 query，
            #   与当前 KV block 中第 j 个 key
            #   的相似度分数。
            #
            # 这就是普通 attention 中 QK^T 的局部块。
            scores = torch.matmul(
                q_block,
                k_block.transpose(-2, -1),
            ) * scale

            # ----------------------------------------------------
            # 13. causal mask
            # ----------------------------------------------------
            #
            # 如果 causal=True，则 query 只能看自己以及自己之前的 key。
            #
            # mask.shape = [Qb, Kb]
            #
            # mask[i, j] = True 表示：
            #   q_positions[i] 可以看 kv_positions[j]
            #
            # 条件是：
            #   kv_position <= q_position
            #
            # 例如：
            #   q_positions = [4,5,6,7]
            #   kv_positions = [4,5,6,7]
            #
            # mask =
            #   q=4: [1,0,0,0]
            #   q=5: [1,1,0,0]
            #   q=6: [1,1,1,0]
            #   q=7: [1,1,1,1]
            #
            # 被 mask 的位置会被设为 -inf，
            # softmax 后概率就变成 0。
            if causal:
                mask = kv_positions.unsqueeze(0) <= q_positions.unsqueeze(1)

                scores = scores.masked_fill(
                    mask.view(1, 1, Qb, Kb) == 0,
                    float("-inf"),
                )

            # ----------------------------------------------------
            # 14. 当前 KV block 内，每个 query row 的最大 score
            # ----------------------------------------------------
            #
            # m_block.shape = [B, H, Qb]
            #
            # 对 scores 的最后一维 Kb 取最大值。
            #
            # m_block[b,h,i] 表示：
            #   当前 Q block 中第 i 个 query，
            #   在当前 KV block 里看到的最大 score。
            #
            # 这个最大值用于 softmax 数值稳定。
            m_block = scores.max(dim=-1).values

            # ----------------------------------------------------
            # 15. valid: 当前 block 对某些 query 是否有可见 key
            # ----------------------------------------------------
            #
            # 在 causal mask 下，可能出现某一行全是 -inf。
            #
            # 例如：
            #   q_position = 2
            #   kv_block positions = [4,5,6,7]
            #
            # 这个 query 对当前 KV block 完全不可见，
            # 那么 m_block = -inf。
            #
            # 如果后面直接计算：
            #
            #   exp(scores - m_block)
            #
            # 会出现：
            #
            #   -inf - (-inf) = nan
            #
            # 所以需要 valid 做保护。
            #
            # valid.shape = [B, H, Qb]
            #
            # valid=True:
            #   当前 query row 在这个 KV block 中至少有一个可见 key。
            valid = torch.isfinite(m_block)

            # ----------------------------------------------------
            # 16. safe_m_block: 避免无效行产生 nan
            # ----------------------------------------------------
            #
            # 对有效行：
            #   safe_m_block = m_block
            #
            # 对无效行：
            #   safe_m_block = 0
            #
            # 无效行后面会强制 p=0，
            # 所以这里设成 0 只是为了避免 nan。
            safe_m_block = torch.where(
                valid,
                m_block,
                torch.zeros_like(m_block),
            )

            # ----------------------------------------------------
            # 17. 计算当前 block 的 exp(scores - m_block)
            # ----------------------------------------------------
            #
            # p.shape = [B, H, Qb, Kb]
            #
            # 注意：
            #   这里的 p 不是最终 softmax 概率。
            #
            # 它只是当前 KV block 内，
            # 基于局部最大值 m_block 做稳定化后的 exp 值。
            #
            # 后面还要和之前的 block 结果做 online softmax 合并。
            p = torch.exp(scores - safe_m_block.unsqueeze(-1))

            # ----------------------------------------------------
            # 18. 对无效 query row，将 p 强制置 0
            # ----------------------------------------------------
            #
            # 如果某个 query 在当前 KV block 中没有可见 key，
            # 那么这个 block 不应该对它的 softmax 分母和输出有任何贡献。
            p = torch.where(
                valid.unsqueeze(-1),
                p,
                torch.zeros_like(p),
            )

            # ----------------------------------------------------
            # 19. 当前 block 的局部分母 l_block
            # ----------------------------------------------------
            #
            # l_block.shape = [B, H, Qb]
            #
            # l_block = sum(exp(scores - m_block))
            #
            # 它是当前 KV block 对 softmax 分母的局部贡献。
            l_block = p.sum(dim=-1)

            # ----------------------------------------------------
            # 20. 当前 block 的局部输出分子 acc_block
            # ----------------------------------------------------
            #
            # acc_block.shape = [B, H, Qb, D]
            #
            # p.shape = [B, H, Qb, Kb]
            # v_block.shape = [B, H, Kb, D]
            #
            # acc_block = p @ v_block
            #
            # 含义：
            #   当前 KV block 内，
            #   每个 query 根据 p 对 V 做加权求和。
            #
            # 注意：
            #   这还不是最终输出。
            #   因为 p 还没有用全局 softmax 分母归一化。
            acc_block = torch.matmul(p, v_block)

            # ====================================================
            # 21. Online Softmax 合并
            # ====================================================
            #
            # 现在我们有：
            #
            # 旧累计状态：
            #   m
            #   l
            #   acc
            #
            # 当前 block 的局部状态：
            #   m_block
            #   l_block
            #   acc_block
            #
            # 需要把二者合并成新的累计状态。
            #
            # 为什么不能直接：
            #   l += l_block
            #   acc += acc_block
            #
            # 因为旧状态和新 block 可能使用了不同的最大值：
            #
            #   旧状态基于 m
            #   新 block 基于 m_block
            #
            # softmax 为了数值稳定，需要统一到新的最大值 m_new。
            # ====================================================

            # ----------------------------------------------------
            # 22. 新的全局最大值 m_new
            # ----------------------------------------------------
            #
            # m.shape = [B, H, Qb]
            # m_block.shape = [B, H, Qb]
            #
            # m_new 表示：
            #   扫描到当前 KV block 之后，
            #   每个 query row 的全局最大 score。
            m_new = torch.maximum(m, m_block)

            # ----------------------------------------------------
            # 23. old_scale: 旧累计状态缩放因子
            # ----------------------------------------------------
            #
            # 旧状态中的 l/acc 是基于旧最大值 m 计算的。
            #
            # 现在最大值变成 m_new，
            # 所以旧状态要乘：
            #
            #   exp(m - m_new)
            #
            # 才能转换到新尺度。
            old_scale = torch.exp(m - m_new)

            # ----------------------------------------------------
            # 24. new_scale: 当前 block 缩放因子
            # ----------------------------------------------------
            #
            # 当前 block 的 l_block/acc_block
            # 是基于 m_block 计算的。
            #
            # 要转换到 m_new 尺度，需要乘：
            #
            #   exp(m_block - m_new)
            new_scale = torch.exp(m_block - m_new)

            # ----------------------------------------------------
            # 25. 避免 -inf 相关计算导致 nan
            # ----------------------------------------------------
            #
            # 初始 m = -inf。
            #
            # 某些情况下：
            #   m = -inf
            #   m_new = -inf
            #
            # 则：
            #   exp(-inf - -inf) = exp(nan) = nan
            #
            # 所以这里把非有限值替换成 0。
            old_scale = torch.where(
                torch.isfinite(old_scale),
                old_scale,
                torch.zeros_like(old_scale),
            )

            new_scale = torch.where(
                torch.isfinite(new_scale),
                new_scale,
                torch.zeros_like(new_scale),
            )

            # ----------------------------------------------------
            # 26. 更新 softmax 分母 l
            # ----------------------------------------------------
            #
            # l_new.shape = [B, H, Qb]
            #
            # 公式：
            #
            #   l_new =
            #       exp(m_old - m_new) * l_old
            #       +
            #       exp(m_block - m_new) * l_block
            #
            # 其中：
            #   old_scale = exp(m_old - m_new)
            #   new_scale = exp(m_block - m_new)
            l_new = old_scale * l + new_scale * l_block

            # ----------------------------------------------------
            # 27. 更新未归一化输出分子 acc
            # ----------------------------------------------------
            #
            # acc_new.shape = [B, H, Qb, D]
            #
            # 公式：
            #
            #   acc_new =
            #       old_scale * acc_old
            #       +
            #       new_scale * acc_block
            #
            # old_scale.shape = [B, H, Qb]
            # acc.shape       = [B, H, Qb, D]
            #
            # 所以需要：
            #   old_scale.unsqueeze(-1)
            #
            # 变成：
            #   [B, H, Qb, 1]
            #
            # 才能和 acc 的最后一维 D 广播相乘。
            acc_new = (
                old_scale.unsqueeze(-1) * acc
                + new_scale.unsqueeze(-1) * acc_block
            )

            # ----------------------------------------------------
            # 28. 用新状态覆盖旧状态
            # ----------------------------------------------------
            #
            # 处理完当前 KV block 后，
            # m/l/acc 就代表：
            #
            #   当前 Q block 已经扫描过从 0 到 kv_end 的所有 K/V block
            #
            # 的累计结果。
            m = m_new
            l = l_new
            acc = acc_new

        # ========================================================
        # 当前 Q block 扫描完所有 K/V block
        # ========================================================
        #
        # 此时：
        #   acc 是完整 softmax 的分子
        #   l 是完整 softmax 的分母
        #
        # 所以：
        #   out_block = acc / l
        #
        # 维度：
        #   acc.shape = [B, H, Qb, D]
        #   l.shape   = [B, H, Qb]
        #
        # l.unsqueeze(-1).shape = [B, H, Qb, 1]
        #
        # out_block.shape = [B, H, Qb, D]
        out_block = acc / l.unsqueeze(-1)

        # --------------------------------------------------------
        # 29. 把当前 Q block 的输出写回总输出 out
        # --------------------------------------------------------
        #
        # 当前 Q block 对应原序列位置：
        #
        #   q_start : q_end
        #
        # 所以写入：
        #
        #   out[:, :, q_start:q_end, :]
        out[:, :, q_start:q_end, :] = out_block

    # ------------------------------------------------------------
    # 30. 返回完整 attention 输出
    # ------------------------------------------------------------
    #
    # out.shape = [B, H, S, D]
    #
    # 数学上等价于：
    #
    #   softmax(QK^T / sqrt(D)) @ V
    #
    # 但中间没有显式保存完整的：
    #
    #   scores.shape = [B, H, S, S]
    #   probs.shape  = [B, H, S, S]
    return out


# ============================================================
# 4. Correctness Check
# ============================================================

def verify_flash_attention_correctness():
    """
    验证教学版 FlashAttention 和 naive attention 输出一致。
    """

    torch.manual_seed(0)

    B = 2
    H = 4
    S = 64
    D = 32

    q = torch.randn(B, H, S, D)
    k = torch.randn(B, H, S, D)
    v = torch.randn(B, H, S, D)

    naive_out = naive_attention(
        q=q,
        k=k,
        v=v,
        causal=True,
    )

    flash_out = flash_attention_forward_tutorial(
        q=q,
        k=k,
        v=v,
        q_block_size=16,
        kv_block_size=16,
        causal=True,
    )

    max_diff = (naive_out - flash_out).abs().max().item()
    mean_diff = (naive_out - flash_out).abs().mean().item()

    print("\n=== FlashAttention Correctness Check ===")
    print(f"naive_out.shape = {tuple(naive_out.shape)}")
    print(f"flash_out.shape = {tuple(flash_out.shape)}")
    print(f"max_diff  = {max_diff:.8f}")
    print(f"mean_diff = {mean_diff:.8f}")


# ============================================================
# 5. Tiny Self-Attention with FlashAttention
# ============================================================

class TinyFlashSelfAttention(nn.Module):
    """
    使用教学版 FlashAttention 的 Self-Attention 层。

    输入：
        x.shape = [B, S, hidden_size]

    内部：
        q/k/v projection
        split heads
        flash_attention_forward_tutorial
        merge heads
        o_proj

    输出：
        out.shape = [B, S, hidden_size]
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        q_block_size: int = 16,
        kv_block_size: int = 16,
        causal: bool = True,
    ):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_block_size = q_block_size
        self.kv_block_size = kv_block_size
        self.causal = causal

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x.shape = [B, S, hidden_size]

        输出：
            x.shape = [B, H, S, D]
        """

        B, S, hidden_size = x.shape

        x = x.view(B, S, self.num_heads, self.head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()

        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x.shape = [B, H, S, D]

        输出：
            x.shape = [B, S, hidden_size]
        """

        B, H, S, D = x.shape

        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, S, H * D)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape = [B, S, hidden_size]
        """

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # q/k/v.shape = [B, H, S, D]
        context = flash_attention_forward_tutorial(
            q=q,
            k=k,
            v=v,
            q_block_size=self.q_block_size,
            kv_block_size=self.kv_block_size,
            causal=self.causal,
        )

        context = self._merge_heads(context)

        out = self.o_proj(context)

        return out


# ============================================================
# 6. Tiny Decoder Block
# ============================================================

class TinyMLP(nn.Module):
    """
    简化 FFN。
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


class TinyFlashDecoderBlock(nn.Module):
    """
    使用 FlashAttention 的 Decoder-only Block。

    Pre-LN：
        x = x + Attention(LN(x))
        x = x + MLP(LN(x))
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        q_block_size: int = 16,
        kv_block_size: int = 16,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = TinyFlashSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            q_block_size=q_block_size,
            kv_block_size=kv_block_size,
            causal=True,
        )

        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = TinyMLP(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================
# 7. Tiny Causal LM with FlashAttention
# ============================================================

class TinyFlashCausalLM(nn.Module):
    """
    一个使用教学版 FlashAttention 的 tiny decoder-only LM。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_layers: int,
        max_seq_len: int,
        q_block_size: int = 16,
        kv_block_size: int = 16,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.blocks = nn.ModuleList([
            TinyFlashDecoderBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
                q_block_size=q_block_size,
                kv_block_size=kv_block_size,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        input_ids.shape = [B, S]
        labels.shape = [B, S]
        """

        B, S = input_ids.shape
        device = input_ids.device

        if S > self.max_seq_len:
            raise ValueError("Sequence length exceeds max_seq_len.")

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

        if labels is None:
            return logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )

        return loss, logits


# ============================================================
# 8. Demo
# ============================================================

def demo_tiny_flash_lm():
    """
    运行一个 Tiny FlashAttention LM forward。
    """

    torch.manual_seed(0)

    model = TinyFlashCausalLM(
        vocab_size=1000,
        hidden_size=128,
        num_heads=8,
        intermediate_size=512,
        num_layers=2,
        max_seq_len=128,
        q_block_size=16,
        kv_block_size=16,
    )

    input_ids = torch.randint(
        low=0,
        high=1000,
        size=(2, 64),
        dtype=torch.long,
    )

    labels = input_ids.clone()

    loss, logits = model(input_ids, labels)

    print("\n=== Tiny FlashAttention LM Demo ===")
    print(f"input_ids.shape = {tuple(input_ids.shape)}")
    print(f"logits.shape    = {tuple(logits.shape)}")
    print(f"loss            = {loss.item():.4f}")


# ============================================================
# 9. Main
# ============================================================

if __name__ == "__main__":
    verify_flash_attention_correctness()
    demo_tiny_flash_lm()