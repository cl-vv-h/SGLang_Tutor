# lora_tutorial.py

"""
============================================================
LoRA / Low-Rank Adaptation 完整教学代码
============================================================

本代码目标：
    1. 从零实现一个 LoRA Linear 层；
    2. 把 LoRA 注入到一个简化 Transformer 的 q_proj / v_proj 中；
    3. 冻结原始模型参数；
    4. 只训练 LoRA 参数；
    5. 展示 LoRA 的训练、推理、合并权重过程。

LoRA 核心思想：

    原始 Linear:
        y = xW

    LoRA 微调:
        y = xW + scaling * xAB

    其中：
        W 是原始大模型权重，冻结不训练。
        A 和 B 是 LoRA 新增的小矩阵，需要训练。
        r 是低秩维度。
        scaling = lora_alpha / r。

数学上：

    ΔW = A @ B

    W' = W + ΔW

LoRA 训练时：
    不更新 W，只更新 A 和 B。

LoRA 推理时：
    可以保持两个分支：
        xW + xAB

    也可以把 ΔW 合并进 W：
        W_merged = W + ΔW
        y = xW_merged

    合并后推理没有额外计算开销。
"""

import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. LoRA 配置
# ============================================================

@dataclass
class LoRAConfig:
    """
    LoRA 配置类。

    r:
        LoRA 的低秩维度。
        r 越大，LoRA 表达能力越强，但参数越多。

    lora_alpha:
        LoRA 缩放因子。
        实际缩放比例是 lora_alpha / r。

    lora_dropout:
        LoRA 分支前的 dropout。
        常用于训练正则化。

    target_modules:
        要注入 LoRA 的模块名称。
        在 LLM 中常见是 ["q_proj", "v_proj"]。
    """

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None


# ============================================================
# 2. LoRALinear
# ============================================================

class LoRALinear(nn.Module):
    """
    一个带 LoRA 的 Linear 层。

    它包装一个原始 nn.Linear，并额外添加两个低秩矩阵：

        lora_A: in_features -> r
        lora_B: r -> out_features

    原始 Linear:

        base_out = xW + b

    LoRA 分支:

        lora_out = xAB * scaling

    最终输出:

        out = base_out + lora_out

    注意：
        原始 Linear 的参数通常冻结。
        训练时只训练 lora_A 和 lora_B。

    PyTorch Linear 的权重形状：

        nn.Linear(in_features, out_features).weight.shape
            = [out_features, in_features]

    因此在合并权重时需要注意矩阵转置。
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        r: int,
        lora_alpha: int,
        lora_dropout: float = 0.0,
    ):
        super().__init__()

        if r <= 0:
            raise ValueError("LoRA rank r must be positive.")

        self.base_linear = base_linear

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.r = r
        self.lora_alpha = lora_alpha

        # LoRA 的缩放系数。
        #
        # 原论文中通常使用：
        #   scaling = alpha / r
        #
        # 这样 r 改变时，LoRA 分支输出的尺度相对稳定。
        self.scaling = lora_alpha / r

        # dropout 只作用在 LoRA 分支上。
        #
        # base linear 分支保持不变。
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        # ------------------------------------------------------------
        # LoRA A 矩阵
        # ------------------------------------------------------------
        #
        # lora_A:
        #   in_features -> r
        #
        # 输入 x 先降维到 r 维。
        self.lora_A = nn.Linear(
            self.in_features,
            r,
            bias=False,
        )

        # ------------------------------------------------------------
        # LoRA B 矩阵
        # ------------------------------------------------------------
        #
        # lora_B:
        #   r -> out_features
        #
        # 再从 r 维升回 out_features。
        self.lora_B = nn.Linear(
            r,
            self.out_features,
            bias=False,
        )

        # LoRA 常用初始化方式：
        #
        # A 用 Kaiming 初始化；
        # B 初始化为 0。
        #
        # 为什么 B 初始化为 0？
        #
        # 初始时：
        #   lora_B(lora_A(x)) = 0
        #
        # 所以模型刚注入 LoRA 时，整体输出和原始模型完全一致：
        #   out = base_out
        #
        # 这样不会破坏原始模型初始行为。
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        # 标记当前 LoRA 是否已经合并进 base weight。
        #
        # 训练时：
        #   merged = False
        #
        # 推理合并后：
        #   merged = True
        self.merged = False

        # 冻结原始 Linear 参数。
        #
        # LoRA 微调的核心就是：
        #   base model 不训练，只训练 lora_A / lora_B。
        for param in self.base_linear.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算。

        x.shape 可以是：

            [batch, in_features]
            [batch, seq_len, in_features]

        nn.Linear 支持最后一维作为 feature 维度，
        所以这里不需要手动 reshape。

        如果 LoRA 已经 merge 到 base weight 中：
            只走 base_linear。

        如果没有 merge：
            out = base_linear(x) + lora_B(lora_A(dropout(x))) * scaling
        """

        # base_out 是原始模型分支。
        #
        # 这部分权重被冻结，不参与训练。
        base_out = self.base_linear(x)

        if self.merged:
            # 如果已经把 LoRA 权重合并进 base_linear.weight，
            # 那就不能再额外加 LoRA 分支，否则会重复计算。
            return base_out

        # LoRA 分支：
        #
        # 1. dropout
        # 2. lora_A 降维
        # 3. lora_B 升维
        # 4. scaling 缩放
        lora_out = self.lora_B(
            self.lora_A(
                self.lora_dropout(x)
            )
        ) * self.scaling

        return base_out + lora_out

    def merge(self):
        """
        将 LoRA 权重合并到 base_linear.weight 中。

        合并前：

            y = xW + scaling * xAB

        合并后：

            W' = W + scaling * AB
            y = xW'

        注意 PyTorch Linear 的 weight 形状：

            base_linear.weight.shape = [out_features, in_features]

        lora_A.weight.shape = [r, in_features]
        lora_B.weight.shape = [out_features, r]

        因此：

            delta_W = lora_B.weight @ lora_A.weight

        delta_W.shape = [out_features, in_features]

        正好可以加到 base_linear.weight 上。
        """

        if self.merged:
            return

        with torch.no_grad():
            delta_w = self.lora_B.weight @ self.lora_A.weight
            delta_w = delta_w * self.scaling

            self.base_linear.weight += delta_w

        self.merged = True

    def unmerge(self):
        """
        将已经 merge 的 LoRA 权重从 base_linear.weight 中减回去。

        训练时通常需要 unmerge，保证 LoRA 分支可以正常训练。

        推理时通常 merge 后就不再 unmerge。
        """

        if not self.merged:
            return

        with torch.no_grad():
            delta_w = self.lora_B.weight @ self.lora_A.weight
            delta_w = delta_w * self.scaling

            self.base_linear.weight -= delta_w

        self.merged = False


# ============================================================
# 3. 一个简化版 Causal Self-Attention
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    简化版 Decoder-only Causal Self-Attention。

    结构：

        x
        ↓
        q_proj, k_proj, v_proj
        ↓
        causal attention
        ↓
        o_proj

    这里我们专门把 q_proj / v_proj 命名出来，
    是为了后面给 q_proj / v_proj 注入 LoRA。
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

        # Attention 中的四个 Linear。
        #
        # LoRA 最常注入 q_proj 和 v_proj。
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将 hidden 拆成多个 attention heads。

        输入:
            x.shape = [B, S, hidden_size]

        输出:
            x.shape = [B, num_heads, S, head_dim]
        """

        B, S, H = x.shape

        x = x.view(B, S, self.num_heads, self.head_dim)
        x = x.transpose(1, 2).contiguous()

        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多头输出合并回 hidden。

        输入:
            x.shape = [B, num_heads, S, head_dim]

        输出:
            x.shape = [B, S, hidden_size]
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

        seq_len = 4 时：

            1 0 0 0
            1 1 0 0
            1 1 1 0
            1 1 1 1

        shape:
            [1, 1, S, S]

        可以广播到 attention score：

            [B, num_heads, S, S]
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

        # 计算 Q/K/V。
        #
        # 如果 q_proj / v_proj 被替换成 LoRALinear，
        # 那么这里自动走 LoRA 分支。
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # attention scores:
        #
        # q.shape = [B, heads, S, head_dim]
        # k.transpose.shape = [B, heads, head_dim, S]
        #
        # scores.shape = [B, heads, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        # causal mask，防止当前 token 看到未来 token。
        mask = self._build_causal_mask(S, device)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)

        context = torch.matmul(attn, v)

        context = self._merge_heads(context)

        out = self.o_proj(context)

        return out


# ============================================================
# 4. FeedForward
# ============================================================

class FeedForward(nn.Module):
    """
    简化 FFN。

    结构：

        hidden_size -> intermediate_size -> hidden_size
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


# ============================================================
# 5. Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    """
    简化 Decoder-only Transformer Block。

    使用 Pre-LN：

        x
        ↓
        norm1
        ↓
        causal self-attention
        ↓
        residual add
        ↓
        norm2
        ↓
        FFN
        ↓
        residual add
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


# ============================================================
# 6. Tiny Causal LM
# ============================================================

class TinyCausalLM(nn.Module):
    """
    一个简化版 Decoder-only Causal LM。

    结构：

        input_ids
            ↓
        token_embedding
            ↓
        position_embedding
            ↓
        TransformerBlock × N
            ↓
        final_norm
            ↓
        lm_head
            ↓
        logits
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
        self.hidden_size = hidden_size
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

        如果 labels 不为空，则返回 loss 和 logits。
        如果 labels 为空，则只返回 logits。
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

        # Causal LM 训练：
        #
        # input:
        #   token0 token1 token2
        #
        # target:
        #   token1 token2 token3
        #
        # 所以 logits 要左移，labels 要右移。
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )

        return loss, logits


# ============================================================
# 7. 注入 LoRA
# ============================================================

def inject_lora(
    model: nn.Module,
    lora_config: LoRAConfig,
):
    """
    给模型中的指定 Linear 模块注入 LoRA。

    这里采用递归遍历模块的方法。

    target_modules 示例：

        ["q_proj", "v_proj"]

    如果某个子模块名字是 q_proj 或 v_proj，并且它是 nn.Linear，
    就把它替换成 LoRALinear。

    替换前：

        block.attn.q_proj = nn.Linear(...)

    替换后：

        block.attn.q_proj = LoRALinear(base_linear=原来的 Linear, ...)
    """

    if lora_config.target_modules is None:
        lora_config.target_modules = ["q_proj", "v_proj"]

    for module_name, module in model.named_children():
        # 如果当前 child 是目标 Linear，则替换。
        if (
            module_name in lora_config.target_modules
            and isinstance(module, nn.Linear)
        ):
            lora_module = LoRALinear(
                base_linear=module,
                r=lora_config.r,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
            )

            setattr(model, module_name, lora_module)

        else:
            # 递归处理子模块。
            inject_lora(module, lora_config)


def freeze_non_lora_parameters(model: nn.Module):
    """
    冻结所有非 LoRA 参数。

    LoRA 微调时，通常只训练：
        lora_A.weight
        lora_B.weight

    其他参数全部冻结。

    注意：
        LoRALinear.__init__ 中已经冻结了 base_linear。
        这里再统一处理一遍，确保只有 LoRA 参数 requires_grad=True。
    """

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def mark_only_lora_as_trainable(model: nn.Module):
    """
    和 freeze_non_lora_parameters 作用相同。
    单独写一个语义更清楚的函数名。
    """

    freeze_non_lora_parameters(model)


def print_trainable_parameters(model: nn.Module):
    """
    打印模型总参数量和可训练参数量。

    用于观察 LoRA 微调节省了多少训练参数。
    """

    total_params = 0
    trainable_params = 0

    for _, param in model.named_parameters():
        numel = param.numel()
        total_params += numel

        if param.requires_grad:
            trainable_params += numel

    ratio = 100 * trainable_params / total_params

    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable ratio:      {ratio:.4f}%")


# ============================================================
# 8. 合并 / 取消合并 LoRA
# ============================================================

def merge_lora_weights(model: nn.Module):
    """
    遍历模型中的所有 LoRALinear，将 LoRA 权重合并到 base weight 中。

    推理部署时常用。

    合并后：
        forward 只走 base_linear；
        不再额外计算 LoRA 分支。
    """

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge()


def unmerge_lora_weights(model: nn.Module):
    """
    遍历模型中的所有 LoRALinear，将合并的 LoRA 权重撤销。

    如果后续还要继续训练 LoRA，需要 unmerge。
    """

    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.unmerge()


# ============================================================
# 9. 构造假数据
# ============================================================

def create_dummy_batch(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
):
    """
    构造一个假的语言模型训练 batch。

    input_ids.shape = [B, S]

    labels 和 input_ids 相同。
    在 causal LM 训练中，模型会内部 shift。
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
# 10. 训练示例
# ============================================================

def train_demo():
    """
    一个最小 LoRA 训练示例。

    注意：
        这里使用随机数据，只是为了演示训练流程。
        真实任务中，需要换成真实数据集。
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # 模型配置
    # ------------------------------------------------------------
    vocab_size = 1000
    hidden_size = 128
    num_heads = 8
    intermediate_size = 512
    num_layers = 2
    max_seq_len = 64

    batch_size = 4
    seq_len = 32

    # ------------------------------------------------------------
    # 1. 创建 base model
    # ------------------------------------------------------------
    model = TinyCausalLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    ).to(device)

    print("\nBefore LoRA injection:")
    print_trainable_parameters(model)

    # ------------------------------------------------------------
    # 2. 注入 LoRA
    # ------------------------------------------------------------
    lora_config = LoRAConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )

    inject_lora(model, lora_config)

    # ------------------------------------------------------------
    # 3. 冻结非 LoRA 参数
    # ------------------------------------------------------------
    mark_only_lora_as_trainable(model)

    print("\nAfter LoRA injection:")
    print_trainable_parameters(model)

    # ------------------------------------------------------------
    # 4. 创建 optimizer
    # ------------------------------------------------------------
    #
    # optimizer 只会收到 requires_grad=True 的参数。
    #
    # 这里也可以直接：
    #   optimizer = AdamW(model.parameters())
    #
    # 但显式过滤更清晰。
    trainable_params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=1e-3,
    )

    model.train()

    # ------------------------------------------------------------
    # 5. 训练循环
    # ------------------------------------------------------------
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
    # 6. 推理前合并 LoRA 权重
    # ------------------------------------------------------------
    #
    # 合并前：
    #   base_linear(x) + lora_B(lora_A(x)) * scaling
    #
    # 合并后：
    #   base_linear_merged(x)
    #
    # 输出应当几乎一致。
    model.eval()

    input_ids, _ = create_dummy_batch(
        batch_size=1,
        seq_len=seq_len,
        vocab_size=vocab_size,
        device=device,
    )

    with torch.no_grad():
        logits_before_merge = model(input_ids)

    merge_lora_weights(model)

    with torch.no_grad():
        logits_after_merge = model(input_ids)

    max_diff = (logits_before_merge - logits_after_merge).abs().max().item()

    print("\nAfter merging LoRA weights:")
    print(f"Max difference before/after merge: {max_diff:.8f}")

    # 如果还要继续训练，可以 unmerge。
    #
    # unmerge_lora_weights(model)


# ============================================================
# 11. 主入口
# ============================================================

if __name__ == "__main__":
    train_demo()