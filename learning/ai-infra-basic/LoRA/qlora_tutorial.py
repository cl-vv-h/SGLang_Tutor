# qlora_tutorial.py

"""
============================================================
QLoRA 教学版源码
============================================================

本代码目标：
    从零实现一个“接近 QLoRA 思想”的教学版本。

核心思想：
    1. Base Model 的 Linear 权重量化成 4-bit NF4-like 格式；
    2. Base Model 权重冻结，不训练；
    3. 在指定 Linear 上注入 LoRA；
    4. 训练时只更新 LoRA A/B；
    5. Forward 时：
        base_out = x @ dequantize(W_4bit)^T
        lora_out = scaling * lora_B(lora_A(x))
        out = base_out + lora_out

注意：
    真实 QLoRA 通常使用 bitsandbytes 的 4-bit Linear。
    这里为了教学可读性，手动实现了一个 NF4-like 量化器。

重要限制：
    1. PyTorch 没有原生 int4 tensor，所以这里用 uint8 存储 0~15 的 code。
    2. 这不会像真实 int4 kernel 那样节省全部计算显存。
    3. 但它能帮助你理解 QLoRA 的 forward / 参数冻结 / LoRA 训练逻辑。
"""

import math
from dataclasses import dataclass
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. QLoRA 配置
# ============================================================

@dataclass
class QLoRAConfig:
    """
    QLoRA 配置。

    r:
        LoRA rank。
        LoRA 的低秩维度。

    lora_alpha:
        LoRA scaling 的分子。
        实际 scaling = lora_alpha / r。

    lora_dropout:
        LoRA 分支前的 dropout。

    target_modules:
        哪些 Linear 要注入 LoRA。
        例如 ["q_proj", "v_proj"]。

    block_size:
        4-bit 量化时，每多少个 weight 共享一个 scale。
        QLoRA 里常见 block size 类似 64。

    use_double_quant:
        是否对 scale 再做一次简化量化。
    """

    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None
    block_size: int = 64
    use_double_quant: bool = True


# ============================================================
# 2. NF4-like Codebook
# ============================================================

def get_nf4_codebook(device: torch.device) -> torch.Tensor:
    """
    返回一个 NF4-like codebook。

    4-bit 可以表示 16 个离散值。
    NF4 的思想是让这 16 个值更贴近正态分布权重的分布。

    这里使用常见的 NF4 近似 codebook：

        codebook.shape = [16]

    每个量化 code 0~15 会映射到 codebook 中的一个浮点数。

    例如：
        q = 0  -> -1.0000
        q = 15 ->  1.0000

    注意：
        真实 bitsandbytes NF4 实现会有更严格的数值细节。
        这里用于教学。
    """

    values = [
        -1.0000,
        -0.6962,
        -0.5251,
        -0.3949,
        -0.2844,
        -0.1848,
        -0.0911,
        0.0000,
        0.0796,
        0.1609,
        0.2461,
        0.3379,
        0.4407,
        0.5626,
        0.7230,
        1.0000,
    ]

    return torch.tensor(values, dtype=torch.float32, device=device)


# ============================================================
# 3. NF4-like 量化函数
# ============================================================

def quantize_nf4_weight(
    weight: torch.Tensor,
    block_size: int = 64,
    use_double_quant: bool = True,
):
    """
    将一个 float weight 矩阵量化成 NF4-like 格式。

    输入：
        weight.shape = [out_features, in_features]

    输出：
        qweight:
            uint8 tensor。
            shape = [out_features, num_blocks, block_size]
            每个值范围是 0~15，表示 NF4 code。

        qscales 或 scales:
            每个 block 的 scale。

        metadata:
            保存原始 shape、padding 信息等。

    为什么要分 block 量化？

        不同 weight block 的数值范围可能不同。
        每个 block 用自己的 scale 可以提升量化精度。

    量化过程：

        1. 把每一行 weight 按 block_size 切块；
        2. 对每个 block 求 absmax 作为 scale；
        3. block 内 weight 除以 scale，归一化到 [-1, 1]；
        4. 每个归一化值找到最近的 NF4 codebook 值；
        5. 保存 code 和 scale。

    维度例子：

        weight.shape = [128, 128]
        block_size = 64

        num_blocks = 128 / 64 = 2

        qweight.shape = [128, 2, 64]
        scales.shape  = [128, 2]
    """

    if weight.dim() != 2:
        raise ValueError("Only 2D Linear weight is supported.")

    device = weight.device
    out_features, in_features = weight.shape

    # 如果 in_features 不能被 block_size 整除，需要 padding。
    pad_len = (block_size - in_features % block_size) % block_size
    padded_in_features = in_features + pad_len

    if pad_len > 0:
        # 在最后一维 padding 0。
        #
        # weight_padded.shape = [out_features, padded_in_features]
        weight_padded = F.pad(weight, pad=(0, pad_len), mode="constant", value=0.0)
    else:
        weight_padded = weight

    # 切成 block。
    #
    # weight_blocks.shape = [out_features, num_blocks, block_size]
    num_blocks = padded_in_features // block_size
    weight_blocks = weight_padded.view(out_features, num_blocks, block_size)

    # 每个 block 一个 scale。
    #
    # scales.shape = [out_features, num_blocks, 1]
    scales = weight_blocks.abs().amax(dim=-1, keepdim=True)

    # 避免除以 0。
    scales = scales.clamp_min(1e-8)

    # 归一化到大约 [-1, 1]。
    #
    # normalized.shape = [out_features, num_blocks, block_size]
    normalized = weight_blocks / scales
    normalized = normalized.clamp(-1.0, 1.0)

    # NF4 codebook。
    #
    # codebook.shape = [16]
    codebook = get_nf4_codebook(device=device)

    # 为每个 normalized 值找到最近的 codebook 项。
    #
    # normalized[..., None].shape:
    #   [out_features, num_blocks, block_size, 1]
    #
    # codebook.view(1,1,1,16).shape:
    #   [1, 1, 1, 16]
    #
    # distance.shape:
    #   [out_features, num_blocks, block_size, 16]
    distance = (
        normalized.unsqueeze(-1)
        - codebook.view(1, 1, 1, 16)
    ).abs()

    # qweight 存储每个 weight 对应的 code。
    #
    # qweight.shape = [out_features, num_blocks, block_size]
    # dtype = uint8
    qweight = distance.argmin(dim=-1).to(torch.uint8)

    # scales 原本 shape = [out_features, num_blocks, 1]
    # 去掉最后一维：
    #
    # scales.shape = [out_features, num_blocks]
    scales = scales.squeeze(-1).to(torch.float32)

    metadata = {
        "out_features": out_features,
        "in_features": in_features,
        "padded_in_features": padded_in_features,
        "pad_len": pad_len,
        "num_blocks": num_blocks,
        "block_size": block_size,
        "use_double_quant": use_double_quant,
    }

    if not use_double_quant:
        return qweight, scales, None, metadata

    # ------------------------------------------------------------
    # 简化版 Double Quantization
    # ------------------------------------------------------------
    #
    # 真实 QLoRA 会进一步量化 scale。
    #
    # 这里为了教学，使用简单的 per-row uint8 量化：
    #
    # scales.shape = [out_features, num_blocks]
    #
    # 对每个 out_feature row：
    #   scale_max = max(scales[row])
    #   qscale = round(scales / scale_max * 255)
    #
    # 保存：
    #   qscales: uint8 [out_features, num_blocks]
    #   scale_max: float32 [out_features, 1]
    #
    # 反量化：
    #   scales_deq = qscales / 255 * scale_max
    scale_max = scales.amax(dim=-1, keepdim=True).clamp_min(1e-8)

    qscales = torch.round(scales / scale_max * 255.0)
    qscales = qscales.clamp(0, 255).to(torch.uint8)

    double_quant_state = {
        "qscales": qscales,
        "scale_max": scale_max.to(torch.float32),
    }

    # 如果使用 double quant，就不直接返回 float scales。
    return qweight, None, double_quant_state, metadata


def dequantize_nf4_weight(
    qweight: torch.Tensor,
    scales: Optional[torch.Tensor],
    double_quant_state: Optional[dict],
    metadata: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    将 NF4-like 权重反量化为 float weight。

    输入：
        qweight.shape = [out_features, num_blocks, block_size]

        如果没有 double quant:
            scales.shape = [out_features, num_blocks]

        如果有 double quant:
            double_quant_state["qscales"].shape = [out_features, num_blocks]
            double_quant_state["scale_max"].shape = [out_features, 1]

    输出：
        weight_deq.shape = [out_features, in_features]

    注意：
        这里返回的是 float32 权重。
        真实 bitsandbytes 会使用更高效的 CUDA kernel，
        不一定显式 materialize 完整 float weight。
    """

    codebook = get_nf4_codebook(device=device)

    # qweight 是 0~15 的 code。
    #
    # codebook[qweight.long()] 会把 code 映射回浮点 codebook 值。
    #
    # normalized_deq.shape = [out_features, num_blocks, block_size]
    normalized_deq = codebook[qweight.long()]

    use_double_quant = metadata["use_double_quant"]

    if use_double_quant:
        qscales = double_quant_state["qscales"].to(device)
        scale_max = double_quant_state["scale_max"].to(device)

        # 反量化 scale。
        #
        # scales_deq.shape = [out_features, num_blocks]
        scales_deq = qscales.float() / 255.0 * scale_max
    else:
        scales_deq = scales.to(device)

    # scales_deq.unsqueeze(-1).shape:
    #   [out_features, num_blocks, 1]
    #
    # weight_blocks.shape:
    #   [out_features, num_blocks, block_size]
    weight_blocks = normalized_deq * scales_deq.unsqueeze(-1)

    out_features = metadata["out_features"]
    in_features = metadata["in_features"]
    padded_in_features = metadata["padded_in_features"]

    # 展平成矩阵：
    #
    # weight_padded.shape = [out_features, padded_in_features]
    weight_padded = weight_blocks.view(out_features, padded_in_features)

    # 去掉 padding：
    #
    # weight_deq.shape = [out_features, in_features]
    weight_deq = weight_padded[:, :in_features]

    return weight_deq


# ============================================================
# 4. QuantizedNF4Linear：冻结的 4-bit Linear
# ============================================================

class QuantizedNF4Linear(nn.Module):
    """
    一个 NF4-like 4-bit 量化 Linear。

    它模拟 QLoRA 中冻结的 base model Linear。

    原始 Linear：

        y = x @ W^T + b

    量化后：

        W_q = quantize_nf4(W)

        forward:
            W_deq = dequantize(W_q)
            y = x @ W_deq^T + b

    注意：
        W_q 是 buffer，不是 nn.Parameter。
        因此它不会被 optimizer 更新。

    输入维度：

        x.shape 可以是：
            [B, in_features]
            [B, S, in_features]

    输出维度：

        out.shape:
            [B, out_features]
            [B, S, out_features]
    """

    def __init__(
        self,
        float_linear: nn.Linear,
        block_size: int = 64,
        use_double_quant: bool = True,
    ):
        super().__init__()

        self.in_features = float_linear.in_features
        self.out_features = float_linear.out_features
        self.has_bias = float_linear.bias is not None

        weight = float_linear.weight.detach().to(torch.float32)

        qweight, scales, double_quant_state, metadata = quantize_nf4_weight(
            weight=weight,
            block_size=block_size,
            use_double_quant=use_double_quant,
        )

        # qweight 是 uint8 code，注册为 buffer。
        #
        # buffer 会随 model.to(device) 移动，
        # 但不会作为可训练参数。
        self.register_buffer("qweight", qweight)

        if scales is not None:
            self.register_buffer("scales", scales)
        else:
            self.scales = None

        if double_quant_state is not None:
            self.register_buffer("qscales", double_quant_state["qscales"])
            self.register_buffer("scale_max", double_quant_state["scale_max"])
        else:
            self.qscales = None
            self.scale_max = None

        if self.has_bias:
            self.register_buffer(
                "bias",
                float_linear.bias.detach().to(torch.float32),
            )
        else:
            self.bias = None

        self.metadata = metadata

    def dequantize_weight(self) -> torch.Tensor:
        """
        反量化得到 float weight。

        返回：
            weight_deq.shape = [out_features, in_features]
        """

        if self.metadata["use_double_quant"]:
            double_quant_state = {
                "qscales": self.qscales,
                "scale_max": self.scale_max,
            }
        else:
            double_quant_state = None

        weight = dequantize_nf4_weight(
            qweight=self.qweight,
            scales=self.scales,
            double_quant_state=double_quant_state,
            metadata=self.metadata,
            device=self.qweight.device,
        )

        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape:
            [B, in_features]
            或 [B, S, in_features]

        weight_deq.shape:
            [out_features, in_features]

        F.linear 会执行：
            out = x @ weight_deq.T + bias

        out.shape:
            [B, out_features]
            或 [B, S, out_features]
        """

        weight_deq = self.dequantize_weight()

        return F.linear(
            input=x,
            weight=weight_deq,
            bias=self.bias,
        )


# ============================================================
# 5. QLoRALinear：4-bit base + LoRA 分支
# ============================================================

class QLoRALinear(nn.Module):
    """
    QLoRA Linear。

    它由两部分组成：

        1. quantized_base:
            冻结的 4-bit base Linear。

        2. LoRA branch:
            可训练的 lora_A 和 lora_B。

    Forward：

        base_out = quantized_base(x)

        lora_out = lora_B(lora_A(dropout(x))) * scaling

        out = base_out + lora_out

    输入：

        x.shape = [B, S, in_features]

    输出：

        out.shape = [B, S, out_features]
    """

    def __init__(
        self,
        float_linear: nn.Linear,
        r: int,
        lora_alpha: int,
        lora_dropout: float,
        block_size: int = 64,
        use_double_quant: bool = True,
    ):
        super().__init__()

        if r <= 0:
            raise ValueError("LoRA rank r must be positive.")

        self.in_features = float_linear.in_features
        self.out_features = float_linear.out_features

        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # 4-bit 冻结 base linear。
        self.quantized_base = QuantizedNF4Linear(
            float_linear=float_linear,
            block_size=block_size,
            use_double_quant=use_double_quant,
        )

        # LoRA dropout。
        self.lora_dropout = nn.Dropout(p=lora_dropout)

        # LoRA A:
        #
        # in_features -> r
        #
        # 输入 x 先降维到 r。
        self.lora_A = nn.Linear(
            self.in_features,
            r,
            bias=False,
        )

        # LoRA B:
        #
        # r -> out_features
        #
        # 再升维回输出维度。
        self.lora_B = nn.Linear(
            r,
            self.out_features,
            bias=False,
        )

        # 初始化：
        #
        # A 随机初始化；
        # B 初始化为 0。
        #
        # 初始时 LoRA 分支输出为 0，
        # 所以 QLoRA 模型初始输出等于量化 base model 输出。
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape:
            [B, S, in_features]

        base_out.shape:
            [B, S, out_features]

        lora_A(x).shape:
            [B, S, r]

        lora_B(lora_A(x)).shape:
            [B, S, out_features]

        out.shape:
            [B, S, out_features]
        """

        base_out = self.quantized_base(x)

        lora_hidden = self.lora_A(
            self.lora_dropout(x)
        )

        lora_out = self.lora_B(lora_hidden) * self.scaling

        return base_out + lora_out


# ============================================================
# 6. Tiny Transformer 模型
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    简化 Decoder-only Causal Self-Attention。

    注意：
        q_proj / k_proj / v_proj / o_proj 都是 nn.Linear。
        后面会被替换成：
            QuantizedNF4Linear 或 QLoRALinear。
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
        输入：
            x.shape = [B, S, hidden_size]

        输出：
            x.shape = [B, num_heads, S, head_dim]
        """

        B, S, H = x.shape

        x = x.view(B, S, self.num_heads, self.head_dim)
        x = x.transpose(1, 2).contiguous()

        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入：
            x.shape = [B, num_heads, S, head_dim]

        输出：
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


class TransformerBlock(nn.Module):
    """
    简化 Decoder-only Transformer Block。

    结构：
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
    一个简化 Decoder-only Causal LM。

    结构：
        token_embedding
        position_embedding
        TransformerBlock × N
        final_norm
        lm_head
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

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        input_ids.shape = [B, S]

        labels.shape = [B, S]

        如果 labels 不为 None：
            返回 loss, logits

        如果 labels 为 None：
            返回 logits
        """

        B, S = input_ids.shape
        device = input_ids.device

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

        # Causal LM loss：
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
# 7. 将模型替换成 QLoRA 结构
# ============================================================

def replace_linear_with_qlora(
    module: nn.Module,
    qlora_config: QLoRAConfig,
):
    """
    递归遍历模型，将所有 nn.Linear 替换为：

        1. QLoRALinear:
            如果模块名在 target_modules 中。

        2. QuantizedNF4Linear:
            如果模块名不在 target_modules 中。

    这更接近 QLoRA：
        整个 base model 的 Linear 都量化；
        只有目标模块额外加 LoRA 分支。

    例如 target_modules = ["q_proj", "v_proj"]：

        q_proj -> QLoRALinear
        v_proj -> QLoRALinear
        k_proj -> QuantizedNF4Linear
        o_proj -> QuantizedNF4Linear
        up_proj -> QuantizedNF4Linear
        down_proj -> QuantizedNF4Linear
        lm_head -> QuantizedNF4Linear
    """

    if qlora_config.target_modules is None:
        qlora_config.target_modules = ["q_proj", "v_proj"]

    for child_name, child_module in list(module.named_children()):

        if isinstance(child_module, nn.Linear):
            if child_name in qlora_config.target_modules:
                new_module = QLoRALinear(
                    float_linear=child_module,
                    r=qlora_config.r,
                    lora_alpha=qlora_config.lora_alpha,
                    lora_dropout=qlora_config.lora_dropout,
                    block_size=qlora_config.block_size,
                    use_double_quant=qlora_config.use_double_quant,
                )
            else:
                new_module = QuantizedNF4Linear(
                    float_linear=child_module,
                    block_size=qlora_config.block_size,
                    use_double_quant=qlora_config.use_double_quant,
                )

            setattr(module, child_name, new_module)

        else:
            replace_linear_with_qlora(child_module, qlora_config)


def freeze_all_non_lora_parameters(model: nn.Module):
    """
    冻结所有非 LoRA 参数。

    QLoRA 中：
        base model 是量化后的 buffer，本身已经不可训练；
        embedding / layernorm 这类参数也通常冻结；
        只训练 LoRA A/B。

    本函数确保：
        只有名字里包含 lora_A 或 lora_B 的参数可训练。
    """

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def print_trainable_parameters(model: nn.Module):
    """
    打印总参数量和可训练参数量。

    注意：
        量化权重是 buffer，不是 parameter；
        因此这里统计的 total parameters 主要包括：
            embedding
            layernorm
            LoRA A/B

        真实模型中，量化权重虽然不是 Parameter，
        但仍然占显存。
    """

    total_params = 0
    trainable_params = 0

    for _, param in model.named_parameters():
        n = param.numel()
        total_params += n

        if param.requires_grad:
            trainable_params += n

    ratio = 100 * trainable_params / max(total_params, 1)

    print(f"Total nn.Parameter count:     {total_params:,}")
    print(f"Trainable parameter count:   {trainable_params:,}")
    print(f"Trainable ratio:             {ratio:.4f}%")


def print_quantized_buffers(model: nn.Module):
    """
    打印量化 buffer 的数量，帮助理解 base model 权重已经不再是 Parameter。
    """

    qweight_count = 0
    qweight_numel = 0

    for module in model.modules():
        if isinstance(module, QuantizedNF4Linear):
            qweight_count += 1
            qweight_numel += module.qweight.numel()

        if isinstance(module, QLoRALinear):
            qweight_count += 1
            qweight_numel += module.quantized_base.qweight.numel()

    print(f"Number of quantized Linear modules: {qweight_count}")
    print(f"Total 4-bit code elements:          {qweight_numel:,}")


# ============================================================
# 8. 构造假数据
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
# 9. QLoRA 训练 Demo
# ============================================================

def train_qlora_demo():
    """
    最小 QLoRA 训练流程。

    流程：
        1. 创建 float base model；
        2. 替换 Linear 为 4-bit Quantized Linear / QLoRALinear；
        3. 冻结非 LoRA 参数；
        4. 只把 LoRA 参数传给 optimizer；
        5. 训练几个 step。
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
    # 1. 创建 float base model
    # ------------------------------------------------------------
    #
    # 在真实场景中，这一步对应加载预训练模型。
    base_model = TinyCausalLM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
    )

    print("\nBefore QLoRA conversion:")
    print_trainable_parameters(base_model)

    # ------------------------------------------------------------
    # 2. 转换为 QLoRA 模型
    # ------------------------------------------------------------
    qlora_config = QLoRAConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        block_size=64,
        use_double_quant=True,
    )

    replace_linear_with_qlora(
        module=base_model,
        qlora_config=qlora_config,
    )

    model = base_model.to(device)

    # ------------------------------------------------------------
    # 3. 冻结非 LoRA 参数
    # ------------------------------------------------------------
    freeze_all_non_lora_parameters(model)

    print("\nAfter QLoRA conversion:")
    print_trainable_parameters(model)
    print_quantized_buffers(model)

    # ------------------------------------------------------------
    # 4. 创建 optimizer
    # ------------------------------------------------------------
    #
    # optimizer 只更新 LoRA A/B。
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
    # 6. 简单推理
    # ------------------------------------------------------------
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
# 10. 主入口
# ============================================================

if __name__ == "__main__":
    train_qlora_demo()