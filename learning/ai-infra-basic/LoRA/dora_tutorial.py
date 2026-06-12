# dora_tutorial.py

"""
============================================================
DoRA / Weight-Decomposed Low-Rank Adaptation 教学版源码
============================================================

DoRA 核心思想：

    普通 LoRA:
        W_eff = W + ΔW
        ΔW = B @ A

    DoRA:
        V = W + ΔW
        direction = V / ||V||
        W_eff = m * direction

其中：

    W:
        原始预训练权重，冻结不训练。

    ΔW:
        LoRA 低秩增量，用来更新 direction。

    m:
        可训练的 magnitude 参数。
        每个输出通道一个 magnitude。

PyTorch Linear 权重形状：

    weight.shape = [out_features, in_features]

所以：

    m.shape = [out_features]

    direction.shape = [out_features, in_features]

    W_eff = m[:, None] * direction

Forward:

    out = F.linear(x, W_eff, bias)

训练时：

    base_linear.weight 冻结
    LoRA A/B 可训练
    magnitude m 可训练

推理时：

    可以将 DoRA 合并成一个普通 Linear 权重：
        W_merged = m * normalize(W + ΔW)

    合并后推理不需要额外 DoRA 分支。
"""

import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. DoRA 配置
# ============================================================

@dataclass
class DoRAConfig:
    """
    DoRA 配置。

    r:
        LoRA rank。
        用于方向更新 ΔW = B @ A。

    lora_alpha:
        LoRA scaling 的分子。
        scaling = lora_alpha / r。

    lora_dropout:
        LoRA 分支前的 dropout。

    target_modules:
        要注入 DoRA 的模块名。
        常见是 ["q_proj", "v_proj"]。
    """

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None


# ============================================================
# 2. DoRALinear
# ============================================================

class DoRALinear(nn.Module):
    """
    带 DoRA 的 Linear。

    它包装一个原始 nn.Linear，并添加：

        1. LoRA A/B:
            用来学习 direction update ΔW。

        2. magnitude:
            每个输出通道一个可训练幅值 m。

    原始 Linear:

        y = x @ W.T + b

    DoRA:

        ΔW = B @ A
        V = W + scaling * ΔW
        direction = V / ||V||_row
        W_eff = magnitude[:, None] * direction

        y = x @ W_eff.T + b

    其中：

        W.shape = [out_features, in_features]
        ΔW.shape = [out_features, in_features]
        magnitude.shape = [out_features]
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        if r <= 0:
            raise ValueError("DoRA rank r must be positive.")

        self.base_linear = base_linear

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.eps = eps

        self.lora_dropout = nn.Dropout(p=lora_dropout)

        # ------------------------------------------------------------
        # LoRA A
        # ------------------------------------------------------------
        #
        # lora_A:
        #   in_features -> r
        #
        # weight shape:
        #   [r, in_features]
        self.lora_A = nn.Linear(
            self.in_features,
            r,
            bias=False,
        )

        # ------------------------------------------------------------
        # LoRA B
        # ------------------------------------------------------------
        #
        # lora_B:
        #   r -> out_features
        #
        # weight shape:
        #   [out_features, r]
        self.lora_B = nn.Linear(
            r,
            self.out_features,
            bias=False,
        )

        # ------------------------------------------------------------
        # magnitude 参数
        # ------------------------------------------------------------
        #
        # magnitude.shape = [out_features]
        #
        # 它表示每个输出通道的权重范数/幅值。
        #
        # 初始化为原始 base weight 每一行的 L2 norm：
        #
        #   magnitude[i] = ||W[i, :]||_2
        #
        # 这样刚开始时，如果 LoRA ΔW = 0：
        #
        #   W_eff = ||W|| * normalize(W) = W
        #
        # 即 DoRA 初始输出和原始模型一致。
        with torch.no_grad():
            weight = base_linear.weight.detach()
            weight_norm = torch.linalg.norm(
                weight,
                ord=2,
                dim=1,
            )

        self.magnitude = nn.Parameter(
            weight_norm.clone()
        )

        # ------------------------------------------------------------
        # 初始化 LoRA
        # ------------------------------------------------------------
        #
        # A 随机初始化；
        # B 初始化为 0。
        #
        # 初始时 ΔW = 0，因此：
        #
        #   V = W
        #   direction = normalize(W)
        #   magnitude = ||W||
        #   W_eff = W
        #
        # 所以 DoRA 插入后不改变原始模型输出。
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # 冻结 base linear。
        for p in self.base_linear.parameters():
            p.requires_grad = False

        # 是否已经 merge。
        self.merged = False

    def compute_delta_weight(self) -> torch.Tensor:
        """
        计算 LoRA 低秩增量 ΔW。

        lora_B.weight.shape:
            [out_features, r]

        lora_A.weight.shape:
            [r, in_features]

        delta_w = B @ A
            shape = [out_features, in_features]

        scaling:
            delta_w *= lora_alpha / r
        """

        delta_w = self.lora_B.weight @ self.lora_A.weight
        delta_w = delta_w * self.scaling
        return delta_w

    def compute_dora_weight(self) -> torch.Tensor:
        """
        计算 DoRA 的有效权重 W_eff。

        base weight:

            W.shape = [out_features, in_features]

        LoRA update:

            delta_w.shape = [out_features, in_features]

        direction source:

            V = W + delta_w
            V.shape = [out_features, in_features]

        row norm:

            row_norm.shape = [out_features, 1]

        direction:

            direction = V / row_norm
            direction.shape = [out_features, in_features]

        magnitude:

            magnitude.shape = [out_features]
            magnitude[:, None].shape = [out_features, 1]

        final:

            W_eff = magnitude[:, None] * direction
            W_eff.shape = [out_features, in_features]
        """

        W = self.base_linear.weight
        delta_w = self.compute_delta_weight()

        V = W + delta_w

        row_norm = torch.linalg.norm(
            V,
            ord=2,
            dim=1,
            keepdim=True,
        ).clamp_min(self.eps)

        direction = V / row_norm

        W_eff = self.magnitude.unsqueeze(1) * direction

        return W_eff

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算。

        输入：

            x.shape = [B, in_features]
            或 [B, S, in_features]

        输出：

            out.shape = [B, out_features]
            或 [B, S, out_features]

        如果已经 merged：
            base_linear.weight 已经被替换成 DoRA 合并后的权重，
            直接走 base_linear。

        如果未 merged：
            动态计算 W_eff，再 F.linear。
        """

        if self.merged:
            return self.base_linear(x)

        W_eff = self.compute_dora_weight()

        return F.linear(
            input=x,
            weight=W_eff,
            bias=self.base_linear.bias,
        )

    def merge(self):
        """
        将 DoRA 权重合并进 base_linear.weight。

        合并前：

            forward 动态计算：
                W_eff = m * normalize(W + ΔW)

        合并后：

            base_linear.weight = W_eff

        然后 forward 直接使用 base_linear。

        注意：
            推理部署时可 merge；
            如果继续训练，需要 unmerge。
        """

        if self.merged:
            return

        with torch.no_grad():
            W_eff = self.compute_dora_weight()
            self.base_linear.weight.copy_(W_eff)

        self.merged = True

    def unmerge(self):
        """
        将 base_linear.weight 恢复到原始权重。

        为了支持 unmerge，需要保存原始权重。

        教学代码为了简洁，没有在初始化时保存 original_weight。
        所以这里给出保护性报错。

        工程实现中可以：
            self.register_buffer("original_weight", base_linear.weight.clone())
        然后 unmerge 时恢复。
        """

        raise NotImplementedError(
            "教学版 DoRALinear 没有保存 original_weight，因此不支持 unmerge。"
        )


# ============================================================
# 3. Tiny Transformer
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    简化 Decoder-only Causal Self-Attention。

    q_proj / v_proj 后续会被替换成 DoRALinear。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
    ):
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
        x.shape = [B, S, hidden_size]

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
            [B, S, hidden_size]
        """

        B, num_heads, S, head_dim = x.shape

        x = x.transpose(1, 2).contiguous()
        x = x.view(B, S, num_heads * head_dim)

        return x

    def _build_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        构造 causal mask。

        mask.shape = [1, 1, S, S]
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
        x.shape = [B, S, hidden_size]
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
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
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

        self.attn = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

        self.norm2 = nn.LayerNorm(hidden_size)

        self.ffn = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

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
# 4. 注入 DoRA
# ============================================================

def inject_dora(
    model: nn.Module,
    config: DoRAConfig,
):
    """
    递归遍历模型，将目标 nn.Linear 替换成 DoRALinear。

    target_modules 例如：

        ["q_proj", "v_proj"]

    替换前：

        block.attn.q_proj = nn.Linear(...)

    替换后：

        block.attn.q_proj = DoRALinear(base_linear=原 Linear, ...)
    """

    if config.target_modules is None:
        config.target_modules = ["q_proj", "v_proj"]

    for child_name, child_module in list(model.named_children()):
        if (
            child_name in config.target_modules
            and isinstance(child_module, nn.Linear)
        ):
            new_module = DoRALinear(
                base_linear=child_module,
                r=config.r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )

            setattr(model, child_name, new_module)

        else:
            inject_dora(child_module, config)


def freeze_non_dora_parameters(model: nn.Module):
    """
    冻结所有非 DoRA 参数。

    只训练：

        lora_A.weight
        lora_B.weight
        magnitude

    其他参数全部冻结。
    """

    for name, param in model.named_parameters():
        if (
            "lora_A" in name
            or "lora_B" in name
            or "magnitude" in name
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False


def merge_dora_weights(model: nn.Module):
    """
    将模型中所有 DoRALinear merge 成普通 Linear 权重。
    """

    for module in model.modules():
        if isinstance(module, DoRALinear):
            module.merge()


def print_trainable_parameters(model: nn.Module):
    """
    打印总参数量和可训练参数量。
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


def print_dora_modules(model: nn.Module):
    """
    打印模型中 DoRA 模块信息。
    """

    idx = 0

    for name, module in model.named_modules():
        if isinstance(module, DoRALinear):
            print(
                f"DoRA module {idx}: {name}, "
                f"in={module.in_features}, out={module.out_features}, r={module.r}, "
                f"magnitude.shape={tuple(module.magnitude.shape)}"
            )
            idx += 1


# ============================================================
# 5. 构造假数据
# ============================================================

def create_dummy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
):
    """
    构造随机语言模型训练数据。

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
# 6. 训练 Demo
# ============================================================

def train_dora_demo():
    """
    最小 DoRA 训练流程。

    流程：

        1. 创建 base model；
        2. 注入 DoRA 到 q_proj / v_proj；
        3. 冻结非 DoRA 参数；
        4. 只训练 LoRA A/B + magnitude；
        5. 训练几个 step；
        6. merge DoRA；
        7. 对比 merge 前后输出差异。
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

    print("\nBefore DoRA injection:")
    print_trainable_parameters(model)

    config = DoRAConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    inject_dora(model, config)
    freeze_non_dora_parameters(model)

    print("\nAfter DoRA injection:")
    print_trainable_parameters(model)
    print_dora_modules(model)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3,
    )

    model.train()

    for step in range(10):
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
        optimizer.step()

        print(f"step={step}, loss={loss.item():.4f}")

    # ------------------------------------------------------------
    # merge 前后输出对比
    # ------------------------------------------------------------
    model.eval()

    input_ids, _ = create_dummy_batch(
        batch_size=1,
        seq_len=8,
        vocab_size=vocab_size,
        device=device,
    )

    with torch.no_grad():
        logits_before_merge = model(input_ids)

    merge_dora_weights(model)

    with torch.no_grad():
        logits_after_merge = model(input_ids)

    max_diff = (logits_before_merge - logits_after_merge).abs().max().item()

    print("\nAfter merging DoRA weights:")
    print(f"input_ids.shape = {tuple(input_ids.shape)}")
    print(f"logits_before_merge.shape = {tuple(logits_before_merge.shape)}")
    print(f"logits_after_merge.shape  = {tuple(logits_after_merge.shape)}")
    print(f"Max difference before/after merge: {max_diff:.8f}")


# ============================================================
# 7. 主入口
# ============================================================

if __name__ == "__main__":
    train_dora_demo()