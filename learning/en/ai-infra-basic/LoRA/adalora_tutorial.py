# adalora_tutorial_v2.py

"""
============================================================
AdaLoRA 教学版源码 v2
============================================================

本版本重点解释：

1. s 是什么？
2. rank component 是什么？
3. 为什么需要 rank_mask，而不是直接把 s[i] 置 0？
4. 怎么判断一个 rank component 是否应该被 mask？
5. 最终 mask 掉多少 rank component 由什么决定？

============================================================
一、普通 LoRA
============================================================

普通 LoRA 的权重更新为：

    ΔW = B @ A

其中：

    A.shape = [r, in_features]
    B.shape = [out_features, r]

所以：

    ΔW.shape = [out_features, in_features]

普通 LoRA 的问题是：

    每个模块的 rank 都是固定的。

例如：

    Layer 0 q_proj: r = 8
    Layer 1 q_proj: r = 8
    Layer 2 q_proj: r = 8

但是不同层的重要性不同，有些层可能需要更多 rank，
有些层可能只需要很少 rank。

============================================================
二、AdaLoRA
============================================================

AdaLoRA 将 LoRA 更新写成类似 SVD 的形式：

    ΔW = P @ diag(s) @ Q

其中：

    P.shape = [out_features, init_r]
    s.shape = [init_r]
    Q.shape = [init_r, in_features]

这里的 s 可以理解为：

    每个低秩方向 / rank component 的强度系数。

第 i 个 rank component 是：

    ΔW_i = s[i] * P[:, i] outer Q[i, :]

完整 ΔW 是：

    ΔW = Σ_i ΔW_i

所以：

    init_r = 有多少个 rank component

============================================================
三、为什么需要 mask？
============================================================

AdaLoRA 中会引入：

    rank_mask.shape = [init_r]

其中：

    rank_mask[i] = 1:
        第 i 个 rank component 启用

    rank_mask[i] = 0:
        第 i 个 rank component 被裁剪 / 屏蔽

forward 中使用：

    effective_s = s * rank_mask

这样如果 mask[i] = 0：

    effective_s[i] = 0

第 i 个 rank component 就不会参与输出。

为什么不直接把 s[i] 设置为 0？

原因：

1. s 是可训练参数，mask 是结构开关，职责不同。
2. 如果直接 s[i] = 0，下一步 optimizer 可能又把它更新成非 0。
3. mask 可以稳定关闭 component，而不破坏 s 原来的值。
4. mask 可以支持动态恢复：mask[i] 可以从 0 变回 1。
5. mask 不改变 P/s/Q 的 shape，不破坏 optimizer state。
6. 训练结束后，再根据 mask 真正压缩 adapter。

============================================================
四、怎么判断是否 mask？
============================================================

不是简单对 s[i] 排序。

更合理的是计算 importance score。

教学版使用：

    importance_i = |s[i] * grad(s[i])|

含义：

    s[i]:
        当前第 i 个 component 的强度

    grad(s[i]):
        loss 对这个 component 的敏感度

    |s[i] * grad(s[i])|:
        近似表示如果关掉这个 component，对 loss 的影响有多大

如果 importance 高：

    关掉它会明显影响 loss，因此保留

如果 importance 低：

    关掉它影响较小，因此可以 mask

============================================================
五、最终 mask 掉多少由什么决定？
============================================================

由 rank budget 决定。

例如：

    num_modules = 4
    init_r = 12
    target_r = 4

初始总 rank component 数：

    4 * 12 = 48

最终目标保留：

    4 * 4 = 16

最终 mask 掉：

    48 - 16 = 32

但是注意：

    不是每个模块都固定保留 4 个。

AdaLoRA 是全局预算分配：

    重要模块可能保留 7 个
    不重要模块可能只保留 1 个

只要总保留数接近 budget 即可。
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. AdaLoRA 配置
# ============================================================

@dataclass
class AdaLoRAConfig:
    """
    AdaLoRA 配置。

    init_r:
        初始 rank。
        AdaLoRA 一开始给每个模块较大的 rank。
        例如 init_r = 12。

    target_r:
        目标平均 rank。
        训练后期总预算会下降到：
            num_modules * target_r
        例如 target_r = 4。

    lora_alpha:
        LoRA scaling 系数的分子。
        scaling = lora_alpha / init_r。

    lora_dropout:
        LoRA 分支前的 dropout。

    target_modules:
        要注入 AdaLoRA 的模块名。
        常见是 ["q_proj", "v_proj"]。

    warmup_steps:
        前多少 step 不裁剪。
        因为刚开始 P/s/Q 还没学到有效信息，
        此时裁剪容易误删重要 component。

    final_prune_step:
        到这个 step 时，rank budget 下降到目标值。
        之后保持目标预算。

    prune_interval:
        每隔多少 step 执行一次 mask 更新。

    min_rank_per_module:
        每个 AdaLoRA 模块至少保留多少个 rank component。
        防止某个模块被完全裁掉。
    """

    init_r: int = 12
    target_r: int = 4
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None

    warmup_steps: int = 5
    final_prune_step: int = 40
    prune_interval: int = 5
    min_rank_per_module: int = 1


# ============================================================
# 2. AdaLoRALinear
# ============================================================

class AdaLoRALinear(nn.Module):
    """
    带 AdaLoRA 的 Linear 层。

    原始 Linear：

        base_out = x @ W.T + b

    AdaLoRA 更新：

        ΔW = P @ diag(s * mask) @ Q

    最终输出：

        out = base_out + scaling * x @ ΔW.T

    但是代码里不会显式构造 ΔW，
    而是分三步计算：

        1. x @ Q.T
        2. 乘以 effective_s = s * rank_mask
        3. 再乘 P.T

    维度：

        x.shape = [B, S, in_features]

        Q.shape = [init_r, in_features]
        Q.T.shape = [in_features, init_r]

        x @ Q.T -> [B, S, init_r]

        effective_s.shape = [init_r]

        [B, S, init_r] * [init_r] -> [B, S, init_r]

        P.shape = [out_features, init_r]
        P.T.shape = [init_r, out_features]

        [B, S, init_r] @ [init_r, out_features]
            -> [B, S, out_features]
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        init_r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        if init_r <= 0:
            raise ValueError("init_r must be positive.")

        self.base_linear = base_linear
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.init_r = init_r

        self.scaling = lora_alpha / init_r
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        # ------------------------------------------------------------
        # P: 输出方向矩阵
        # ------------------------------------------------------------
        #
        # P.shape = [out_features, init_r]
        #
        # 第 i 列 P[:, i] 表示第 i 个 rank component
        # 在输出空间中的方向。
        self.P = nn.Parameter(
            torch.empty(self.out_features, init_r)
        )

        # ------------------------------------------------------------
        # s: 每个 rank component 的强度系数
        # ------------------------------------------------------------
        #
        # s.shape = [init_r]
        #
        # 为什么是 [init_r]？
        #
        # 因为 s 是 r 个标量系数：
        #
        #   s = [s0, s1, ..., s_{r-1}]
        #
        # 它相当于 diag(s) 的对角线。
        #
        # 代码中不显式构造 diag(s)，而是直接对中间张量
        # [B, S, r] 做逐维缩放。
        self.s = nn.Parameter(
            torch.empty(init_r)
        )

        # ------------------------------------------------------------
        # Q: 输入方向矩阵
        # ------------------------------------------------------------
        #
        # Q.shape = [init_r, in_features]
        #
        # 第 i 行 Q[i, :] 表示第 i 个 rank component
        # 在输入空间中的方向。
        self.Q = nn.Parameter(
            torch.empty(init_r, self.in_features)
        )

        # ------------------------------------------------------------
        # rank_mask: 结构开关，不是可训练参数
        # ------------------------------------------------------------
        #
        # rank_mask.shape = [init_r]
        #
        # rank_mask[i] = 1:
        #     第 i 个 rank component 启用。
        #
        # rank_mask[i] = 0:
        #     第 i 个 rank component 被 mask。
        #
        # 为什么用 buffer？
        #
        # buffer 会跟随模型移动到 GPU，
        # 但不会被 optimizer 更新。
        self.register_buffer(
            "rank_mask",
            torch.ones(init_r)
        )

        # ------------------------------------------------------------
        # 初始化
        # ------------------------------------------------------------
        #
        # P 和 Q 用 Kaiming 初始化。
        nn.init.kaiming_uniform_(self.P, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.Q, a=math.sqrt(5))

        # s 初始化为一个小值。
        #
        # 如果 s 初始化为全 0，初始 LoRA 分支输出为 0，
        # 这类似普通 LoRA 中 B 初始化为 0。
        #
        # 但为了让 importance score 更容易观察，
        # 教学版使用 1e-3。
        nn.init.constant_(self.s, 1e-3)

        # 冻结 base linear。
        #
        # AdaLoRA 微调只训练 P/s/Q。
        for p in self.base_linear.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算。

        输入：
            x.shape = [B, S, in_features]
            或 [B, in_features]

        输出：
            out.shape = [B, S, out_features]
            或 [B, out_features]
        """

        # 原始 base model 输出。
        #
        # base_linear 参数被冻结，不参与训练。
        base_out = self.base_linear(x)

        # LoRA 分支前 dropout。
        x_dropped = self.lora_dropout(x)

        # ------------------------------------------------------------
        # Step 1: 输入投影到 init_r 个低秩方向
        # ------------------------------------------------------------
        #
        # x_dropped.shape = [..., in_features]
        # Q.T.shape = [in_features, init_r]
        #
        # lora_hidden.shape = [..., init_r]
        lora_hidden = torch.matmul(
            x_dropped,
            self.Q.t()
        )

        # ------------------------------------------------------------
        # Step 2: 用 s 和 mask 控制每个 rank component
        # ------------------------------------------------------------
        #
        # s 是可训练强度。
        # rank_mask 是结构开关。
        #
        # effective_s[i] = s[i] * rank_mask[i]
        #
        # 如果 rank_mask[i] = 0，
        # 第 i 个 rank component 被关闭。
        effective_s = self.s * self.rank_mask

        # lora_hidden.shape = [..., init_r]
        # effective_s.shape = [init_r]
        #
        # PyTorch 自动 broadcast。
        lora_hidden = lora_hidden * effective_s

        # ------------------------------------------------------------
        # Step 3: 从低秩空间映射回输出空间
        # ------------------------------------------------------------
        #
        # lora_hidden.shape = [..., init_r]
        # P.T.shape = [init_r, out_features]
        #
        # lora_out.shape = [..., out_features]
        lora_out = torch.matmul(
            lora_hidden,
            self.P.t()
        )

        return base_out + lora_out * self.scaling

    def effective_rank(self) -> int:
        """
        当前实际启用的 rank component 数量。

        等于 rank_mask 中 1 的数量。
        """

        return int(self.rank_mask.sum().item())

    def importance_score(self) -> torch.Tensor:
        """
        计算每个 rank component 的重要性分数。

        教学版使用一阶泰勒近似：

            importance_i = |s[i] * grad(s[i])|

        为什么不用单纯 |s[i]|？

            因为 s[i] 大，只能说明当前数值强；
            但不一定说明 loss 对它敏感。

        为什么引入 grad(s[i])？

            grad(s[i]) 表示 loss 对第 i 个 component 的敏感程度。

        一阶近似：

            如果把 s[i] 关掉，相当于：
                Δs[i] = -s[i]

            loss 变化：
                ΔL ≈ grad(s[i]) * Δs[i]

            所以影响大小：
                |ΔL| ≈ |s[i] * grad(s[i])|

        返回：
            score.shape = [init_r]
        """

        if self.s.grad is None:
            return torch.zeros_like(self.s.detach())

        return (self.s.detach() * self.s.grad.detach()).abs()

    def set_rank_mask(self, keep_indices: torch.Tensor):
        """
        根据要保留的 indices 设置 mask。

        keep_indices:
            shape = [num_keep]

        效果：
            keep_indices 中的位置 mask=1
            其他位置 mask=0
        """

        new_mask = torch.zeros_like(self.rank_mask)
        new_mask[keep_indices] = 1.0
        self.rank_mask.copy_(new_mask)

    def delta_weight(self) -> torch.Tensor:
        """
        显式计算当前有效 ΔW。

        注意：
            forward 中不会显式构造 ΔW。
            这个函数主要用于理解或导出。

        ΔW = P @ diag(s * mask) @ Q

        等价写法：

            P_scaled = P * effective_s[None, :]
            ΔW = P_scaled @ Q

        维度：

            P.shape = [out_features, init_r]
            effective_s.shape = [init_r]

            P_scaled.shape = [out_features, init_r]

            Q.shape = [init_r, in_features]

            ΔW.shape = [out_features, in_features]
        """

        effective_s = self.s * self.rank_mask
        P_scaled = self.P * effective_s.unsqueeze(0)
        dw = P_scaled @ self.Q
        dw = dw * self.scaling
        return dw


# ============================================================
# 3. AdaLoRA Rank Budget Scheduler
# ============================================================

class AdaLoRABudgetScheduler:
    """
    AdaLoRA rank 预算调度器。

    它负责：

        1. 统计所有 AdaLoRALinear 模块；
        2. 根据 step 计算当前总 rank budget；
        3. 计算所有 rank component 的 importance score；
        4. 全局保留 top-k；
        5. 其他 component 设置 mask=0。

    关键点：

        不是每个模块都保留 target_r 个。
        而是在全局预算下，谁重要谁保留。

    例如：

        有 4 个模块，每个 init_r=12，target_r=4。

        初始总 rank = 48。
        最终总 rank = 16。

        最终可能是：
            module0: 7
            module1: 3
            module2: 5
            module3: 1

        总数仍然是 16。
    """

    def __init__(
        self,
        model: nn.Module,
        init_r: int,
        target_r: int,
        warmup_steps: int,
        final_prune_step: int,
        prune_interval: int,
        min_rank_per_module: int = 1,
    ):
        self.model = model
        self.init_r = init_r
        self.target_r = target_r
        self.warmup_steps = warmup_steps
        self.final_prune_step = final_prune_step
        self.prune_interval = prune_interval
        self.min_rank_per_module = min_rank_per_module

        self.modules = self._find_adalora_modules()
        self.num_modules = len(self.modules)

        if self.num_modules == 0:
            raise ValueError("No AdaLoRALinear modules found.")

        self.initial_total_rank = self.num_modules * init_r
        self.target_total_rank = self.num_modules * target_r

    def _find_adalora_modules(self) -> List[AdaLoRALinear]:
        """
        找到模型里的所有 AdaLoRALinear 模块。
        """

        result = []
        for m in self.model.modules():
            if isinstance(m, AdaLoRALinear):
                result.append(m)
        return result

    def current_budget(self, step: int) -> int:
        """
        计算当前 step 下应该保留多少个 rank component。

        规则：

        1. warmup 前：
            保留全部 rank。

        2. warmup 到 final_prune_step：
            线性减少总预算。

        3. final_prune_step 后：
            保持 target_total_rank。
        """

        if step < self.warmup_steps:
            return self.initial_total_rank

        if step >= self.final_prune_step:
            return self.target_total_rank

        progress = (
            (step - self.warmup_steps)
            / max(1, self.final_prune_step - self.warmup_steps)
        )

        budget = (
            self.initial_total_rank
            - progress * (self.initial_total_rank - self.target_total_rank)
        )

        return int(round(budget))

    def should_prune(self, step: int) -> bool:
        """
        是否在当前 step 更新 mask。

        warmup 阶段不裁剪；
        之后每隔 prune_interval 执行一次裁剪。
        """

        if step < self.warmup_steps:
            return False

        return step % self.prune_interval == 0

    def step(self, step: int):
        """
        在训练循环中调用。

        注意：
            需要在 loss.backward() 之后调用，
            因为 importance_score 依赖 s.grad。

        推荐顺序：

            loss.backward()
            scheduler.step(step)
            optimizer.step()
        """

        if not self.should_prune(step):
            return

        budget = self.current_budget(step)

        # 至少保证每个模块保留 min_rank_per_module。
        min_total = self.num_modules * self.min_rank_per_module
        budget = max(budget, min_total)

        # ------------------------------------------------------------
        # Step 1: 收集所有 rank component 的重要性
        # ------------------------------------------------------------

        all_scores = []
        mapping: List[Tuple[int, int]] = []

        for module_idx, module in enumerate(self.modules):
            score = module.importance_score()

            for rank_idx in range(module.init_r):
                all_scores.append(score[rank_idx])
                mapping.append((module_idx, rank_idx))

        all_scores_tensor = torch.stack(all_scores)

        # ------------------------------------------------------------
        # Step 2: 先给每个模块保底 min_rank_per_module
        # ------------------------------------------------------------
        #
        # 为什么要保底？
        #
        # 如果完全全局 top-k，某些模块可能一个 rank 都没有。
        # 教学版里为了结构稳定，给每个模块至少保留 1 个。
        keep_per_module: Dict[int, List[int]] = {
            i: [] for i in range(self.num_modules)
        }

        already_kept_global_indices = set()

        for module_idx, module in enumerate(self.modules):
            score = module.importance_score()

            k = min(self.min_rank_per_module, module.init_r)

            _, local_top = torch.topk(score, k=k)

            for rank_idx in local_top.tolist():
                keep_per_module[module_idx].append(rank_idx)

                # 找到它在全局 mapping 里的 index。
                global_idx = module_idx * self.init_r + rank_idx
                already_kept_global_indices.add(global_idx)

        remaining_budget = budget - len(already_kept_global_indices)

        # ------------------------------------------------------------
        # Step 3: 剩余预算按全局 importance top-k 分配
        # ------------------------------------------------------------

        if remaining_budget > 0:
            # 为已经保底保留的 component 设置为 -inf，
            # 避免重复选择。
            scores_for_global = all_scores_tensor.clone()

            for idx in already_kept_global_indices:
                scores_for_global[idx] = -float("inf")

            remaining_budget = min(
                remaining_budget,
                scores_for_global.numel() - len(already_kept_global_indices)
            )

            if remaining_budget > 0:
                _, top_global_indices = torch.topk(
                    scores_for_global,
                    k=remaining_budget,
                )

                for global_idx in top_global_indices.tolist():
                    module_idx, rank_idx = mapping[global_idx]
                    keep_per_module[module_idx].append(rank_idx)

        # ------------------------------------------------------------
        # Step 4: 更新每个模块的 rank_mask
        # ------------------------------------------------------------

        for module_idx, keep_list in keep_per_module.items():
            module = self.modules[module_idx]

            # 去重。
            keep_list = sorted(set(keep_list))

            keep_indices = torch.tensor(
                keep_list,
                dtype=torch.long,
                device=module.rank_mask.device,
            )

            module.set_rank_mask(keep_indices)

        self.print_status(step, budget)

    def print_status(self, step: int, budget: int):
        """
        打印当前每个模块的 effective rank。
        """

        ranks = [m.effective_rank() for m in self.modules]
        print(
            f"[AdaLoRA mask update] step={step}, "
            f"budget={budget}, effective_ranks={ranks}"
        )


# ============================================================
# 4. Tiny Transformer Components
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    简化版 Causal Self-Attention。

    q_proj 和 v_proj 后续会被注入 AdaLoRA。
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape = [B, S, H]

        return:
            [B, num_heads, S, head_dim]
        """

        B, S, H = x.shape

        x = x.view(B, S, self.num_heads, self.head_dim)
        x = x.transpose(1, 2).contiguous()

        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape = [B, num_heads, S, head_dim]

        return:
            [B, S, H]
        """

        B, num_heads, S, head_dim = x.shape

        x = x.transpose(1, 2).contiguous()
        x = x.view(B, S, num_heads * head_dim)

        return x

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        causal mask:

        S=4 时：

            1 0 0 0
            1 1 0 0
            1 1 1 0
            1 1 1 1

        return shape:
            [1, 1, S, S]
        """

        mask = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        )

        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape = [B, S, H]
        """

        B, S, H = x.shape
        device = x.device

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        mask = self._build_causal_mask(S, device)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v)

        context = self._merge_heads(context)

        out = self.o_proj(context)

        return out


class FeedForward(nn.Module):
    """
    简化 FFN。

    hidden_size -> intermediate_size -> hidden_size
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


class TransformerBlock(nn.Module):
    """
    Decoder-only Transformer Block。

    Pre-LN 结构：

        x = x + Attention(LN(x))
        x = x + FFN(LN(x))
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(hidden_size, num_heads)

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TinyCausalLM(nn.Module):
    """
    简化 Decoder-only Causal LM。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_layers: int,
        max_seq_len: int,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
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
            raise ValueError("S exceeds max_seq_len.")

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

        # Causal LM:
        #
        # logits[:, t] 预测 labels[:, t+1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )

        return loss, logits


# ============================================================
# 5. 注入 AdaLoRA
# ============================================================

def inject_adalora(model: nn.Module, config: AdaLoRAConfig):
    """
    递归遍历模型，将目标 Linear 替换成 AdaLoRALinear。

    例如：

        target_modules = ["q_proj", "v_proj"]

    替换前：

        block.attn.q_proj = nn.Linear(...)

    替换后：

        block.attn.q_proj = AdaLoRALinear(base_linear=原来的 Linear)
    """

    if config.target_modules is None:
        config.target_modules = ["q_proj", "v_proj"]

    for child_name, child_module in list(model.named_children()):
        if (
            child_name in config.target_modules
            and isinstance(child_module, nn.Linear)
        ):
            new_module = AdaLoRALinear(
                base_linear=child_module,
                init_r=config.init_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )

            setattr(model, child_name, new_module)

        else:
            inject_adalora(child_module, config)


def freeze_non_adalora_parameters(model: nn.Module):
    """
    冻结所有非 AdaLoRA 参数。

    只训练：
        P
        s
        Q

    不训练：
        base model
        embedding
        layernorm
        lm_head
    """

    for name, param in model.named_parameters():
        if ".P" in name or ".s" in name or ".Q" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def print_trainable_parameters(model: nn.Module):
    """
    打印参数量。
    """

    total = 0
    trainable = 0

    for _, p in model.named_parameters():
        n = p.numel()
        total += n

        if p.requires_grad:
            trainable += n

    ratio = 100 * trainable / total

    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Trainable ratio:      {ratio:.4f}%")


def print_adalora_ranks(model: nn.Module):
    """
    打印每个 AdaLoRA 模块当前有效 rank。
    """

    ranks = []

    for m in model.modules():
        if isinstance(m, AdaLoRALinear):
            ranks.append(m.effective_rank())

    print(f"Current AdaLoRA effective ranks: {ranks}")


# ============================================================
# 6. 数据
# ============================================================

def create_dummy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
):
    """
    构造随机语言模型数据。

    input_ids.shape = [B, S]
    labels.shape = [B, S]
    """

    input_ids = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=device,
    )

    labels = input_ids.clone()

    return input_ids, labels


# ============================================================
# 7. 训练 Demo
# ============================================================

def train_demo():
    """
    AdaLoRA 训练流程。

    重点看训练循环中的顺序：

        loss.backward()
        scheduler.step(step)
        optimizer.step()

    为什么 scheduler.step 放在 optimizer.step 前？

        因为 importance score 使用的是当前 backward 得到的 s.grad。

        如果先 optimizer.step()，参数已经更新，
        但我们想基于当前 step 的梯度判断重要性。
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = 1000
    hidden_size = 128
    num_heads = 8
    intermediate_size = 512
    num_layers = 2
    max_seq_len = 64

    batch_size = 4
    seq_len = 32

    model = TinyCausalLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    ).to(device)

    print("\nBefore AdaLoRA injection:")
    print_trainable_parameters(model)

    config = AdaLoRAConfig(
        init_r=12,
        target_r=4,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        warmup_steps=5,
        final_prune_step=40,
        prune_interval=5,
        min_rank_per_module=1,
    )

    inject_adalora(model, config)
    freeze_non_adalora_parameters(model)

    print("\nAfter AdaLoRA injection:")
    print_trainable_parameters(model)
    print_adalora_ranks(model)

    scheduler = AdaLoRABudgetScheduler(
        model=model,
        init_r=config.init_r,
        target_r=config.target_r,
        warmup_steps=config.warmup_steps,
        final_prune_step=config.final_prune_step,
        prune_interval=config.prune_interval,
        min_rank_per_module=config.min_rank_per_module,
    )

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )

    model.train()

    total_steps = 50

    for step in range(total_steps):
        input_ids, labels = create_dummy_batch(
            batch_size=batch_size,
            seq_len=seq_len,
            vocab_size=vocab_size,
            device=device,
        )

        loss, logits = model(
            input_ids=input_ids,
            labels=labels,
        )

        optimizer.zero_grad()
        loss.backward()

        # 关键：
        # 在 optimizer.step 前，根据当前梯度更新 mask。
        scheduler.step(step)

        optimizer.step()

        if step % 5 == 0:
            print(f"step={step}, loss={loss.item():.4f}")

    print("\nFinal AdaLoRA ranks:")
    print_adalora_ranks(model)

    model.eval()

    input_ids, _ = create_dummy_batch(
        batch_size=1,
        seq_len=8,
        vocab_size=vocab_size,
        device=device,
    )

    with torch.no_grad():
        logits = model(input_ids)

    print("\nInference:")
    print(f"input_ids.shape = {tuple(input_ids.shape)}")
    print(f"logits.shape    = {tuple(logits.shape)}")


# ============================================================
# 8. 主入口
# ============================================================

if __name__ == "__main__":
    train_demo()