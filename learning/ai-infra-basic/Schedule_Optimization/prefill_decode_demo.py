# prefill_decode_demo.py

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. KV Cache 数据结构
# ============================================================

@dataclass
class LayerKVCache:
    """
    保存某一层 Transformer Block 的 KV Cache。

    对 decoder-only Transformer 来说，每一层 self-attention 都会产生 K 和 V。

    在普通训练 / 普通 forward 中，每次输入完整序列，K/V 临时计算完就丢弃。

    在推理中，我们希望把历史 token 的 K/V 保存下来。
    下一个 decode step 只需要为新 token 计算 K/V，
    然后把新 K/V 拼到历史 K/V 后面。

    k.shape = [batch_size, num_heads, cached_len, head_dim]
    v.shape = [batch_size, num_heads, cached_len, head_dim]

    cached_len 表示当前这个 cache 里已经保存了多少个历史 token。
    """

    k: torch.Tensor
    v: torch.Tensor


# ============================================================
# 2. 多头 Causal Self-Attention，支持 KV Cache
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    GPT-like 模型中的多头因果自注意力。

    这个模块同时支持三种模式：

    1. 普通 full forward:
        输入完整序列，不传 past_kv，不返回 cache。
        常用于训练。

    2. Prefill:
        输入完整 prompt，不传 past_kv，但返回当前 prompt 的 K/V cache。
        这是推理的第一阶段。

    3. Decode:
        每次只输入一个新 token，同时传入 past_kv。
        模型只计算新 token 的 Q/K/V，
        然后把新 K/V 追加到 past_kv 中。
        这是推理的第二阶段。

    输入 x:
        x.shape = [B, T, d_model]

    其中:
        B = batch_size
        T = 当前输入 token 数

    在 prefill 阶段:
        T = prompt_len

    在 decode 阶段:
        T = 1
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = nn.Dropout(dropout)

        # 这里为了清晰，分别定义 Wq/Wk/Wv。
        # 真实大模型实现里，经常会合并成一个 qkv_proj：
        # nn.Linear(d_model, 3 * d_model)
        #
        # 但分开写更容易理解。
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 多头拼接后的输出投影。
        self.w_o = nn.Linear(d_model, d_model)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        把 [B, T, d_model] 拆成多头格式 [B, H, T, D]。

        B = batch_size
        T = 当前输入长度
        H = num_heads
        D = head_dim

        例如:
            x.shape = [2, 5, 128]
            num_heads = 8
            head_dim = 16

        reshape:
            [2, 5, 8, 16]

        transpose:
            [2, 8, 5, 16]
        """

        B, T, _ = x.shape

        x = x.view(B, T, self.num_heads, self.head_dim)

        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        把多头格式 [B, H, T, D] 合并回 [B, T, d_model]。
        """

        B, H, T, D = x.shape

        x = x.transpose(1, 2).contiguous()

        x = x.view(B, T, H * D)

        return x

    def build_causal_mask(
        self,
        query_len: int,
        key_len: int,
        device: torch.device,
        past_len: int,
    ) -> torch.Tensor:
        """
        构造 causal mask。

        causal mask 的目标:
            每个 query 位置只能看它自己和它之前的 key 位置，
            不能看未来位置。

        对 decoder-only 模型来说，attention 分数形状是:

            scores.shape = [B, H, query_len, key_len]

        mask 需要能 broadcast 到这个 shape:

            mask.shape = [1, 1, query_len, key_len]

        为什么需要 past_len?

        因为 decode 时，我们不是从位置 0 开始计算，
        而是当前输入 token 接在 past cache 后面。

        举例 1: Prefill 阶段
            prompt_len = 4
            query_len = 4
            key_len = 4
            past_len = 0

            mask:
                1 0 0 0
                1 1 0 0
                1 1 1 0
                1 1 1 1

        举例 2: Decode 阶段
            历史已经有 4 个 token
            当前输入 1 个新 token

            query_len = 1
            key_len = 5
            past_len = 4

            当前新 token 的绝对位置是 4。
            它可以看 key 位置 0,1,2,3,4。

            mask:
                1 1 1 1 1

        举例 3: 如果一次 decode 多个新 token
            past_len = 4
            query_len = 3
            key_len = 7

            当前 query 的绝对位置分别是:
                4, 5, 6

            第一个新 token 可以看 key 0..4
            第二个新 token 可以看 key 0..5
            第三个新 token 可以看 key 0..6

            mask:
                1 1 1 1 1 0 0
                1 1 1 1 1 1 0
                1 1 1 1 1 1 1
        """

        # query 的绝对位置。
        #
        # prefill:
        #   past_len = 0
        #   query_pos = [0, 1, 2, ..., query_len - 1]
        #
        # decode:
        #   past_len = 已缓存历史长度
        #   query_pos = [past_len, past_len + 1, ...]
        query_pos = torch.arange(
            past_len,
            past_len + query_len,
            device=device,
        ).unsqueeze(-1)

        # key 的绝对位置。
        #
        # key_len = past_len + query_len
        #
        # key_pos = [0, 1, 2, ..., key_len - 1]
        key_pos = torch.arange(
            0,
            key_len,
            device=device,
        ).unsqueeze(0)

        # query 只能看 key_pos <= query_pos 的位置。
        #
        # mask.shape = [query_len, key_len]
        mask = key_pos <= query_pos

        # 增加 batch 和 head 维度:
        #
        # [query_len, key_len]
        # -> [1, 1, query_len, key_len]
        mask = mask.unsqueeze(0).unsqueeze(0)

        return mask

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[LayerKVCache] = None,
        use_cache: bool = False,
        debug_name: str = "",
    ) -> Tuple[torch.Tensor, Optional[LayerKVCache]]:
        """
        参数:
            x:
                当前输入 hidden states
                shape = [B, T, d_model]

            past_kv:
                历史 KV Cache。
                如果是 prefill 阶段，一般为 None。
                如果是 decode 阶段，一般不为 None。

            use_cache:
                是否返回新的 KV Cache。

            debug_name:
                用于打印调试信息。

        返回:
            out:
                attention 输出
                shape = [B, T, d_model]

            new_kv:
                如果 use_cache=True，返回更新后的 KV Cache。
                否则返回 None。
        """

        B, T, _ = x.shape
        device = x.device

        # ============================================================
        # 1. 当前输入生成 Q/K/V
        # ============================================================

        # q/k/v shape:
        #   [B, T, d_model]
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # 拆成多头:
        #
        # q/k/v shape:
        #   [B, H, T, head_dim]
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # ============================================================
        # 2. 判断是否存在历史 KV Cache
        # ============================================================

        if past_kv is None:
            # prefill 阶段:
            #   past_len = 0
            #   当前 key/value 就是完整 prompt 的 key/value
            past_len = 0

            full_k = k
            full_v = v
        else:
            # decode 阶段:
            #   past_kv.k 保存历史 token 的 key
            #   past_kv.v 保存历史 token 的 value
            #
            # past_kv.k.shape:
            #   [B, H, past_len, head_dim]
            past_len = past_kv.k.size(2)

            # 把历史 K/V 和当前 token 的 K/V 拼接起来。
            #
            # full_k.shape:
            #   [B, H, past_len + T, head_dim]
            #
            # decode 时通常 T = 1，
            # 所以 full_k 的长度每次增加 1。
            full_k = torch.cat([past_kv.k, k], dim=2)
            full_v = torch.cat([past_kv.v, v], dim=2)

        key_len = full_k.size(2)

        # ============================================================
        # 3. 计算 attention score
        # ============================================================

        # q.shape:
        #   [B, H, T, head_dim]
        #
        # full_k.transpose(-2, -1).shape:
        #   [B, H, head_dim, key_len]
        #
        # scores.shape:
        #   [B, H, T, key_len]
        #
        # prefill:
        #   T = prompt_len
        #   key_len = prompt_len
        #   scores = [B, H, prompt_len, prompt_len]
        #
        # decode:
        #   T = 1
        #   key_len = history_len + 1
        #   scores = [B, H, 1, history_len + 1]
        scores = torch.matmul(q, full_k.transpose(-2, -1))

        scores = scores / math.sqrt(self.head_dim)

        # ============================================================
        # 4. 构造并应用 causal mask
        # ============================================================

        causal_mask = self.build_causal_mask(
            query_len=T,
            key_len=key_len,
            device=device,
            past_len=past_len,
        )

        # mask 中 False 的位置表示不能关注。
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        # ============================================================
        # 5. softmax 得到注意力权重
        # ============================================================

        # attn_weights.shape:
        #   [B, H, T, key_len]
        attn_weights = F.softmax(scores, dim=-1)

        attn_weights = self.dropout(attn_weights)

        # ============================================================
        # 6. 注意力权重加权 V
        # ============================================================

        # attn_weights.shape:
        #   [B, H, T, key_len]
        #
        # full_v.shape:
        #   [B, H, key_len, head_dim]
        #
        # context.shape:
        #   [B, H, T, head_dim]
        context = torch.matmul(attn_weights, full_v)

        # ============================================================
        # 7. 多头合并 + 输出投影
        # ============================================================

        # [B, H, T, head_dim] -> [B, T, d_model]
        context = self.combine_heads(context)

        out = self.w_o(context)

        # ============================================================
        # 8. 是否返回新的 KV Cache
        # ============================================================

        if use_cache:
            # new_kv 保存 full_k/full_v。
            #
            # prefill 后:
            #   cache_len = prompt_len
            #
            # decode 每一步后:
            #   cache_len = cache_len + 1
            new_kv = LayerKVCache(k=full_k, v=full_v)
        else:
            new_kv = None

        # ============================================================
        # 9. 调试打印
        # ============================================================

        if debug_name:
            print(f"\n[{debug_name}] Attention Debug")
            print("x.shape:          ", tuple(x.shape))
            print("q.shape:          ", tuple(q.shape))
            print("current k.shape:  ", tuple(k.shape))
            print("past_len:         ", past_len)
            print("full_k.shape:     ", tuple(full_k.shape))
            print("scores.shape:     ", tuple(scores.shape))
            print("causal_mask.shape:", tuple(causal_mask.shape))
            print("out.shape:        ", tuple(out.shape))

        return out, new_kv


# ============================================================
# 3. Feed Forward Network
# ============================================================

class FeedForward(nn.Module):
    """
    Transformer Block 中的 FFN。

    结构:
        Linear(d_model -> d_ff)
        GELU
        Linear(d_ff -> d_model)

    注意:
        FFN 不负责 token 之间的信息交互。
        token 之间的信息交互由 self-attention 完成。

    FFN 是对每个 token 位置独立作用的。
    """

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = [B, T, d_model]

        x = self.fc1(x)

        x = F.gelu(x)

        x = self.fc2(x)

        return x


# ============================================================
# 4. Decoder Block，支持 KV Cache
# ============================================================

class DecoderBlock(nn.Module):
    """
    GPT-like Decoder Block。

    结构采用 Pre-LN 形式:

        x
        ↓
        LayerNorm
        ↓
        Causal Self-Attention
        ↓
        Residual Add
        ↓
        LayerNorm
        ↓
        FFN
        ↓
        Residual Add

    为什么使用 Pre-LN?
        现代大模型中 Pre-LN 更常见，训练更稳定。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[LayerKVCache] = None,
        use_cache: bool = False,
        debug_name: str = "",
    ) -> Tuple[torch.Tensor, Optional[LayerKVCache]]:
        """
        x.shape = [B, T, d_model]

        past_kv:
            当前层对应的历史 KV Cache。

        use_cache:
            是否返回当前层更新后的 KV Cache。
        """

        # ============================================================
        # 1. Attention 子层
        # ============================================================

        # Pre-LN:
        #   先做 LayerNorm，再进入 Attention。
        normed_x = self.norm1(x)

        attn_out, new_kv = self.attn(
            x=normed_x,
            past_kv=past_kv,
            use_cache=use_cache,
            debug_name=debug_name,
        )

        # 残差连接:
        #   x + attention 输出
        x = x + attn_out

        # ============================================================
        # 2. FFN 子层
        # ============================================================

        normed_x = self.norm2(x)

        ffn_out = self.ffn(normed_x)

        # 残差连接:
        #   x + FFN 输出
        x = x + ffn_out

        return x, new_kv


# ============================================================
# 5. Decoder-only Transformer，支持 prefill / decode
# ============================================================

class TinyCausalLM(nn.Module):
    """
    一个最小 GPT-like Causal Language Model。

    支持三种使用方式:

    1. forward_no_cache:
        普通完整序列前向，不使用 KV Cache。
        可用于训练理解。

    2. prefill:
        输入完整 prompt，计算 logits 和所有层的 KV Cache。
        用于推理第一阶段。

    3. decode:
        输入一个新 token，传入历史 KV Cache，
        输出新 logits 和更新后的 KV Cache。
        用于推理后续阶段。
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_heads: int = 8,
        d_ff: int = 512,
        num_layers: int = 2,
        max_seq_len: int = 1024,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # token embedding:
        #   把 token id 映射为向量。
        #
        # input_ids.shape = [B, T]
        # token_emb.shape = [B, T, d_model]
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # learned position embedding:
        #   这里为了更直观，使用可训练位置编码。
        #
        # position_ids.shape = [B, T]
        # pos_emb.shape     = [B, T, d_model]
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # lm_head:
        #   把 hidden state 映射到词表 logits。
        #
        # hidden.shape = [B, T, d_model]
        # logits.shape = [B, T, vocab_size]
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def build_position_ids(
        self,
        input_ids: torch.Tensor,
        past_len: int = 0,
    ) -> torch.Tensor:
        """
        构造 position_ids。

        input_ids.shape = [B, T]

        prefill:
            past_len = 0
            input_ids 是完整 prompt
            position_ids = [0, 1, 2, ..., T-1]

        decode:
            past_len = 历史 cache 长度
            input_ids 通常只有一个新 token
            position_ids = [past_len]

        举例:
            prompt_len = 4

            prefill:
                input_ids = [a, b, c, d]
                position_ids = [0, 1, 2, 3]

            decode 第一步:
                past_len = 4
                input_ids = [token1]
                position_ids = [4]

            decode 第二步:
                past_len = 5
                input_ids = [token2]
                position_ids = [5]
        """

        B, T = input_ids.shape
        device = input_ids.device

        position_ids = torch.arange(
            past_len,
            past_len + T,
            device=device,
            dtype=torch.long,
        )

        # [T] -> [1, T] -> [B, T]
        position_ids = position_ids.unsqueeze(0).expand(B, T)

        return position_ids

    def embed(
        self,
        input_ids: torch.Tensor,
        past_len: int = 0,
    ) -> torch.Tensor:
        """
        token embedding + position embedding。

        input_ids.shape = [B, T]

        返回:
            x.shape = [B, T, d_model]
        """

        # token_emb.shape = [B, T, d_model]
        token_emb = self.token_embedding(input_ids)

        # position_ids.shape = [B, T]
        position_ids = self.build_position_ids(
            input_ids=input_ids,
            past_len=past_len,
        )

        # pos_emb.shape = [B, T, d_model]
        pos_emb = self.position_embedding(position_ids)

        # token embedding + position embedding
        x = token_emb + pos_emb

        return x

    def forward_no_cache(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        普通完整序列前向，不使用 KV Cache。

        这个函数适合用来理解训练时的 full forward。

        input_ids.shape = [B, T]

        返回:
            logits.shape = [B, T, vocab_size]

        注意:
            虽然不使用 KV Cache，但 Attention 内部仍然会计算完整序列的 K/V。
            只是这些 K/V 不会被保存到下一步。
        """

        # past_len = 0，因为没有历史 cache
        x = self.embed(input_ids, past_len=0)

        for block in self.blocks:
            x, _ = block(
                x=x,
                past_kv=None,
                use_cache=False,
            )

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits

    def prefill(
        self,
        input_ids: torch.Tensor,
        debug: bool = False,
    ) -> Tuple[torch.Tensor, List[LayerKVCache]]:
        """
        Prefill 阶段。

        作用:
            1. 输入完整 prompt。
            2. 一次性计算 prompt 所有 token 的 hidden states。
            3. 为每一层生成 KV Cache。
            4. 返回 logits 和 KV Cache。

        input_ids.shape = [B, prompt_len]

        返回:
            logits.shape = [B, prompt_len, vocab_size]

            kv_cache:
                一个 list，长度 = num_layers。
                kv_cache[i] 是第 i 层的 KV Cache。

                kv_cache[i].k.shape =
                    [B, H, prompt_len, head_dim]

                kv_cache[i].v.shape =
                    [B, H, prompt_len, head_dim]
        """

        # prefill 没有历史 cache，所以 past_len = 0
        x = self.embed(input_ids, past_len=0)

        new_cache: List[LayerKVCache] = []

        for layer_idx, block in enumerate(self.blocks):
            debug_name = f"prefill_layer_{layer_idx}" if debug else ""

            # past_kv=None:
            #   表示没有历史缓存。
            #
            # use_cache=True:
            #   表示返回当前层的 K/V cache。
            x, layer_cache = block(
                x=x,
                past_kv=None,
                use_cache=True,
                debug_name=debug_name,
            )

            assert layer_cache is not None

            new_cache.append(layer_cache)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits, new_cache

    def decode(
        self,
        input_ids: torch.Tensor,
        past_cache: List[LayerKVCache],
        debug: bool = False,
    ) -> Tuple[torch.Tensor, List[LayerKVCache]]:
        """
        Decode 阶段。

        作用:
            1. 输入当前新 token，通常长度为 1。
            2. 复用 past_cache 中的历史 K/V。
            3. 每一层只为新 token 计算 Q/K/V。
            4. 把新 token 的 K/V append 到 cache。
            5. 返回当前 token 的 logits 和更新后的 cache。

        input_ids.shape = [B, 1]
            通常 decode 阶段一次只输入一个 token。

        past_cache:
            prefill 或上一步 decode 得到的 KV Cache。
            长度 = num_layers。

        返回:
            logits.shape = [B, 1, vocab_size]

            new_cache:
                更新后的 KV Cache。
                每一层的 cache_len 比输入 past_cache 多 1。
        """

        assert len(past_cache) == self.num_layers

        # 从第 0 层 cache 中取 cached_len。
        #
        # 所有层的 cache_len 应该一致。
        past_len = past_cache[0].k.size(2)

        # decode 时 position_id 应该从 past_len 开始。
        #
        # 比如 prompt_len=4，则第一个 decode token 的位置是 4。
        x = self.embed(input_ids, past_len=past_len)

        new_cache: List[LayerKVCache] = []

        for layer_idx, block in enumerate(self.blocks):
            debug_name = f"decode_layer_{layer_idx}" if debug else ""

            # 当前层使用对应层的 past_kv。
            #
            # 注意:
            #   每一层都有自己的 KV Cache，
            #   不能混用其他层的 cache。
            x, layer_cache = block(
                x=x,
                past_kv=past_cache[layer_idx],
                use_cache=True,
                debug_name=debug_name,
            )

            assert layer_cache is not None

            new_cache.append(layer_cache)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits, new_cache


# ============================================================
# 6. Greedy 生成：先 Prefill，再循环 Decode
# ============================================================

@torch.no_grad()
def greedy_generate(
    model: TinyCausalLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
    eos_id: Optional[int] = None,
    debug: bool = False,
) -> torch.Tensor:
    """
    使用 KV Cache 的自回归生成。

    推理流程:

        1. Prefill:
            输入完整 prompt。
            得到 logits 和 kv_cache。

        2. 取 prefill 最后一个位置的 logits:
            logits[:, -1, :]

            这个位置用于预测 prompt 后面的第一个新 token。

        3. 选出 next_token。

        4. Decode 循环:
            每次只把 next_token 输入模型。
            模型复用 kv_cache。
            输出下一个 logits。
            继续采样。

    参数:
        prompt_ids:
            shape = [B, prompt_len]

        max_new_tokens:
            最多生成多少个新 token。

        eos_id:
            如果生成 EOS，则提前停止。

        debug:
            是否打印 attention shape。

    返回:
        generated:
            shape = [B, prompt_len + generated_len]
    """

    model.eval()

    # ============================================================
    # 1. Prefill
    # ============================================================

    # logits.shape = [B, prompt_len, vocab_size]
    #
    # kv_cache:
    #   list 长度 = num_layers
    #
    #   每层:
    #       k/v.shape = [B, H, prompt_len, head_dim]
    logits, kv_cache = model.prefill(prompt_ids, debug=debug)

    # 当前完整序列从 prompt 开始。
    generated = prompt_ids

    # ============================================================
    # 2. 根据 prefill 最后一个位置预测第一个新 token
    # ============================================================

    # logits[:, -1, :] 表示:
    #   prompt 最后一个 token 位置的输出分布。
    #
    # 它用于预测 prompt 后的下一个 token。
    next_token_logits = logits[:, -1, :]

    # greedy decoding:
    #   直接选择 logits 最大的 token。
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

    # 把第一个新 token 拼到 generated 后面。
    generated = torch.cat([generated, next_token], dim=1)

    if debug:
        print("\nAfter prefill:")
        print("prompt_ids.shape:", tuple(prompt_ids.shape))
        print("generated.shape: ", tuple(generated.shape))
        print("next_token:      ", next_token.tolist())
        print("cache_len:       ", kv_cache[0].k.size(2))

    # 如果只需要生成 1 个 token，直接返回。
    if max_new_tokens == 1:
        return generated

    # ============================================================
    # 3. Decode 循环
    # ============================================================

    # 已经生成了 1 个新 token，所以从 1 开始。
    for step in range(1, max_new_tokens):
        if eos_id is not None:
            # 如果 batch 中所有样本的上一个 token 都是 EOS，就停止。
            if torch.all(next_token.squeeze(-1) == eos_id):
                break

        # decode 输入只包含上一步生成的新 token。
        #
        # input_ids.shape = [B, 1]
        decode_input = next_token

        # logits.shape = [B, 1, vocab_size]
        #
        # kv_cache 会被更新:
        #   cache_len 每次 +1
        logits, kv_cache = model.decode(
            input_ids=decode_input,
            past_cache=kv_cache,
            debug=debug,
        )

        # 当前 decode 输出只有一个位置，
        # 所以取 logits[:, -1, :] 或 logits[:, 0, :] 都可以。
        next_token_logits = logits[:, -1, :]

        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

        if debug:
            print(f"\nAfter decode step {step}:")
            print("decode_input.shape:", tuple(decode_input.shape))
            print("generated.shape:   ", tuple(generated.shape))
            print("next_token:        ", next_token.tolist())
            print("cache_len:         ", kv_cache[0].k.size(2))

    return generated


# ============================================================
# 7. 对比：不用 KV Cache 的低效生成
# ============================================================

@torch.no_grad()
def greedy_generate_without_cache(
    model: TinyCausalLM,
    prompt_ids: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """
    不使用 KV Cache 的生成。

    每一步都把完整 generated 序列重新输入模型。

    这能帮助你对比理解:
        为什么 prefill/decode + KV Cache 更高效。

    缺点:
        第 t 步会重新计算前面所有 token 的 K/V。
    """

    model.eval()

    generated = prompt_ids

    for step in range(max_new_tokens):
        # 每一步都输入完整序列:
        #
        # step 0:
        #   [prompt]
        #
        # step 1:
        #   [prompt, token1]
        #
        # step 2:
        #   [prompt, token1, token2]
        logits = model.forward_no_cache(generated)

        next_token_logits = logits[:, -1, :]

        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=1)

    return generated


# ============================================================
# 8. 主函数：运行 demo
# ============================================================

def main():
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 这里构造一个很小的 toy model，方便你看 shape。
    #
    # 真实大模型中:
    #   vocab_size 可能是 100k+
    #   d_model 可能是 4096 / 8192
    #   num_layers 可能是 32 / 80
    #   num_heads 可能是 32 / 64
    vocab_size = 100
    d_model = 128
    num_heads = 8
    d_ff = 512
    num_layers = 2
    max_seq_len = 64

    model = TinyCausalLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        dropout=0.0,
    ).to(device)

    # 构造一个 prompt。
    #
    # shape = [B, prompt_len]
    #
    # 这里 B=1, prompt_len=5。
    prompt_ids = torch.tensor(
        [[10, 20, 30, 40, 50]],
        dtype=torch.long,
        device=device,
    )

    print("========== Prefill + Decode with KV Cache ==========")

    generated = greedy_generate(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=4,
        eos_id=None,
        debug=True,
    )

    print("\nFinal generated with cache:")
    print(generated.tolist())

    print("\n========== Generate without KV Cache ==========")

    generated_no_cache = greedy_generate_without_cache(
        model=model,
        prompt_ids=prompt_ids,
        max_new_tokens=4,
    )

    print("\nFinal generated without cache:")
    print(generated_no_cache.tolist())


if __name__ == "__main__":
    main()