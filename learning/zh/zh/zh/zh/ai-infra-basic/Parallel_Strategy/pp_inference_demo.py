# pp_autoregressive_demo.py

"""
教学目标：
    用完整代码理解 Pipeline Parallelism，流水线并行，在自回归推理中的执行方式。

本代码实现的是一个 2-stage Pipeline Parallel 的 Decoder-only Transformer。

模型被拆成两个 stage：

    rank 0 / stage 0:
        token_embedding
        position_embedding
        TransformerBlock 0 ~ split_layer - 1

    rank 1 / stage 1:
        TransformerBlock split_layer ~ num_layers - 1
        final_norm
        lm_head

自回归生成流程：

    初始:
        rank 0 持有 prompt_ids，作为 generated。

    每一步:
        rank 0:
            1. 用当前 generated 执行 stage0；
            2. 得到 hidden states；
            3. 把 hidden states 发送给 rank 1。

        rank 1:
            1. 接收 hidden states；
            2. 执行 stage1；
            3. 得到 logits；
            4. 取最后一个位置 logits；
            5. argmax 得到 next_token；
            6. 把 next_token 发回 rank 0。

        rank 0:
            1. 接收 next_token；
            2. 拼接到 generated 后面；
            3. 进入下一轮。

注意：
    这个版本没有实现 KV Cache。
    因此每一轮都会重新计算完整 generated 序列。
    这样效率不高，但最适合理解 PP 自回归推理的数据流。

真实大模型推理中会进一步优化为：
    prefill:
        每个 stage 处理完整 prompt，并建立本 stage 的 KV Cache。

    decode:
        每个 stage 每步只处理一个新 token，并复用自己的 KV Cache。
"""

import os
import math
import argparse
from typing import Optional, Tuple

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

    运行方式：

        torchrun --nproc_per_node=2 pp_autoregressive_demo.py

    torchrun 会自动给每个进程注入环境变量：

        RANK:
            当前进程的全局编号。

        WORLD_SIZE:
            当前分布式任务一共有多少个进程。

        LOCAL_RANK:
            当前进程在本机上的编号，通常用于绑定 GPU。

    本教学代码只实现 2-stage PP：

        rank 0 = stage 0
        rank 1 = stage 1

    因此 world_size 必须等于 2。
    """

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if use_cpu:
        # CPU 模拟时使用 gloo 后端。
        backend = "gloo"
        device = torch.device("cpu")
    else:
        # GPU 场景通常使用 nccl 后端。
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
    """
    退出前销毁进程组。
    """
    dist.destroy_process_group()


# ============================================================
# 2. Transformer 基础模块
# ============================================================

class CausalSelfAttention(nn.Module):
    """
    Decoder-only Transformer 中的因果自注意力。

    这里是普通单卡版 Attention。
    也就是说，每个 pipeline stage 内部的 block 不做 TP。

    如果要组合 TP + PP，可以把这里的 Linear 替换为：
        ColumnParallelLinear / RowParallelLinear

    当前 Attention 结构：

        x
        ↓
        q_proj / k_proj / v_proj
        ↓
        reshape 成多头
        ↓
        causal self-attention
        ↓
        o_proj
        ↓
        out
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        把 hidden 拆成多头。

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
        把多头输出合并回 hidden。

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

        seq_len = 5 时：

            1 0 0 0 0
            1 1 0 0 0
            1 1 1 0 0
            1 1 1 1 0
            1 1 1 1 1

        返回:
            mask.shape = [1, 1, seq_len, seq_len]

        可以广播到:
            scores.shape = [B, num_heads, seq_len, seq_len]
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
        输入:
            x.shape = [B, S, hidden_size]

        输出:
            out.shape = [B, S, hidden_size]
        """

        B, S, H = x.shape
        device = x.device

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        # scores.shape = [B, num_heads, S, S]
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)

        mask = self._build_causal_mask(S, device)
        scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # context.shape = [B, num_heads, S, head_dim]
        context = torch.matmul(attn, v)

        context = self._merge_heads(context)

        out = self.o_proj(context)

        return out


class FeedForward(nn.Module):
    """
    Transformer Block 中的 FFN。

    结构:
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
    一个普通 Decoder-only Transformer Block。

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
        dropout: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = FeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入:
            x.shape = [B, S, hidden_size]

        输出:
            x.shape = [B, S, hidden_size]
        """

        x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x


# ============================================================
# 3. Pipeline Stage 0
# ============================================================

class PipelineStage0(nn.Module):
    """
    Pipeline Stage 0。

    rank 0 持有这个模块。

    它负责：
        1. token embedding；
        2. position embedding；
        3. 前半部分 Transformer Blocks。

    输入:
        input_ids.shape = [B, S]

    输出:
        hidden.shape = [B, S, hidden_size]

    注意：
        Stage 0 不包含 lm_head。
        它不会直接输出 logits。
        它只把中间 hidden states 发送给 Stage 1。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        max_seq_len: int,
        num_stage_layers: int,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            hidden_size,
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_stage_layers)
        ])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids.shape = [B, S]

        返回:
            hidden.shape = [B, S, hidden_size]
        """

        B, S = input_ids.shape
        device = input_ids.device

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

        return x


# ============================================================
# 4. Pipeline Stage 1
# ============================================================

class PipelineStage1(nn.Module):
    """
    Pipeline Stage 1。

    rank 1 持有这个模块。

    它负责：
        1. 后半部分 Transformer Blocks；
        2. final_norm；
        3. lm_head。

    输入:
        hidden.shape = [B, S, hidden_size]

    输出:
        logits.shape = [B, S, vocab_size]

    Stage 1 是最后一个 stage，所以它负责产生 next_token。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        intermediate_size: int,
        num_stage_layers: int,
    ):
        super().__init__()

        self.blocks = nn.ModuleList([
            TransformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_stage_layers)
        ])

        self.final_norm = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        hidden.shape = [B, S, hidden_size]

        返回:
            logits.shape = [B, S, vocab_size]
        """

        x = hidden

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


# ============================================================
# 5. Tensor 通信工具
# ============================================================

def send_tensor_shape(tensor: torch.Tensor, dst: int):
    """
    发送 tensor 的 shape。

    为什么要单独发 shape？

    因为接收方在调用 dist.recv 之前，
    必须先创建一个大小正确的接收 buffer。

    本教学代码为了通用性，每次先发送 shape，再发送实际 tensor。

    真实高性能系统中，shape 往往由调度器提前知道，
    不会每次都额外通信。
    """

    shape = torch.tensor(
        list(tensor.shape),
        dtype=torch.long,
        device=tensor.device,
    )

    ndim = torch.tensor(
        [shape.numel()],
        dtype=torch.long,
        device=tensor.device,
    )

    dist.send(ndim, dst=dst)
    dist.send(shape, dst=dst)


def recv_tensor_shape(
    src: int,
    device: torch.device,
) -> Tuple[int, ...]:
    """
    接收 tensor shape。

    返回:
        Python tuple，例如 (B, S, hidden_size)
    """

    ndim = torch.empty(
        1,
        dtype=torch.long,
        device=device,
    )

    dist.recv(ndim, src=src)

    shape = torch.empty(
        int(ndim.item()),
        dtype=torch.long,
        device=device,
    )

    dist.recv(shape, src=src)

    return tuple(int(x.item()) for x in shape)


def send_tensor(tensor: torch.Tensor, dst: int):
    """
    发送一个 tensor。

    流程:
        1. 先发送 shape；
        2. 再发送实际 tensor 数据。
    """

    send_tensor_shape(tensor, dst=dst)

    dist.send(tensor.contiguous(), dst=dst)


def recv_tensor(
    src: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    接收一个 tensor。

    流程:
        1. 先接收 shape；
        2. 根据 shape 创建 buffer；
        3. 接收实际 tensor 数据。
    """

    shape = recv_tensor_shape(src=src, device=device)

    tensor = torch.empty(
        shape,
        dtype=dtype,
        device=device,
    )

    dist.recv(tensor, src=src)

    return tensor


def send_token(token: torch.Tensor, dst: int):
    """
    发送 next_token。

    token.shape = [B, 1]

    在 2-stage PP 自回归生成中：
        rank 1 是最后一个 stage，负责生成 next_token；
        rank 0 是第一个 stage，下一轮需要这个 next_token 做 embedding；
        所以 rank 1 必须把 next_token 发回 rank 0。
    """

    dist.send(token.contiguous(), dst=dst)


def recv_token(
    src: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """
    接收 next_token。

    返回:
        token.shape = [B, 1]
    """

    token = torch.empty(
        batch_size,
        1,
        dtype=torch.long,
        device=device,
    )

    dist.recv(token, src=src)

    return token


def send_stop_signal(dst: int, device: torch.device):
    """
    发送停止信号。

    在本代码中：
        rank 0 决定生成结束后，会告诉 rank 1 停止循环。

    stop_flag:
        1 表示继续
        0 表示停止
    """

    flag = torch.tensor(
        [0],
        dtype=torch.long,
        device=device,
    )

    dist.send(flag, dst=dst)


def send_continue_signal(dst: int, device: torch.device):
    """
    发送继续信号。

    rank 0 在每一步开始前告诉 rank 1：
        这一轮还有 hidden 要接收。
    """

    flag = torch.tensor(
        [1],
        dtype=torch.long,
        device=device,
    )

    dist.send(flag, dst=dst)


def recv_continue_or_stop(
    src: int,
    device: torch.device,
) -> bool:
    """
    rank 1 接收来自 rank 0 的控制信号。

    返回:
        True:
            继续本轮生成，需要接收 hidden。

        False:
            停止生成，退出循环。
    """

    flag = torch.empty(
        1,
        dtype=torch.long,
        device=device,
    )

    dist.recv(flag, src=src)

    return bool(flag.item() == 1)


# ============================================================
# 6. 自回归式 PP 推理
# ============================================================

@torch.no_grad()
def pipeline_greedy_generate_no_cache(
    rank: int,
    device: torch.device,
    stage0: Optional[PipelineStage0],
    stage1: Optional[PipelineStage1],
    prompt_ids: Optional[torch.Tensor],
    max_new_tokens: int,
    eos_id: Optional[int] = None,
) -> Optional[torch.Tensor]:
    """
    2-stage Pipeline Parallel 自回归生成。

    这是无 KV Cache 的教学版本。

    rank 0:
        1. 保存 generated；
        2. 每一步将 generated 输入 Stage 0；
        3. 得到 hidden；
        4. 发送 hidden 到 rank 1；
        5. 接收 rank 1 返回的 next_token；
        6. 拼接 next_token；
        7. 重复直到 max_new_tokens 或 EOS。

    rank 1:
        1. 每一步等待 rank 0 的控制信号；
        2. 如果继续，则接收 hidden；
        3. 执行 Stage 1；
        4. 得到 logits；
        5. 取最后一个位置，得到 next_token；
        6. 把 next_token 发回 rank 0；
        7. 重复直到收到停止信号。

    参数:
        prompt_ids:
            只有 rank 0 拥有真实 prompt。
            rank 1 传 None 即可。

        max_new_tokens:
            最多生成多少个新 token。

        eos_id:
            可选。如果所有 batch 样本都生成 EOS，则提前停止。

    返回:
        rank 0 返回 generated。
        rank 1 返回 None。
    """

    if rank == 0:
        assert stage0 is not None
        assert prompt_ids is not None

        # rank 0 保存完整 generated。
        #
        # shape = [B, current_len]
        generated = prompt_ids.clone()

        batch_size = generated.size(0)

        for step in range(max_new_tokens):
            # 告诉 rank 1：本轮继续，有 hidden 要接收。
            send_continue_signal(dst=1, device=device)

            # --------------------------------------------------------
            # 1. Stage 0 前向
            # --------------------------------------------------------
            #
            # 因为本教学版本没有 KV Cache，
            # 所以每一轮都重新处理完整 generated。
            #
            # hidden.shape = [B, current_len, hidden_size]
            hidden = stage0(generated)

            print(
                f"[rank 0] step={step}, "
                f"generated shape={tuple(generated.shape)}, "
                f"send hidden shape={tuple(hidden.shape)}",
                flush=True,
            )

            # --------------------------------------------------------
            # 2. 发送 hidden 给 rank 1
            # --------------------------------------------------------
            send_tensor(hidden, dst=1)

            # --------------------------------------------------------
            # 3. 接收 rank 1 返回的 next_token
            # --------------------------------------------------------
            next_token = recv_token(
                src=1,
                batch_size=batch_size,
                device=device,
            )

            print(
                f"[rank 0] step={step}, "
                f"recv next_token={next_token.squeeze(-1).tolist()}",
                flush=True,
            )

            # --------------------------------------------------------
            # 4. 拼接 next_token
            # --------------------------------------------------------
            generated = torch.cat([generated, next_token], dim=1)

            # --------------------------------------------------------
            # 5. EOS 提前停止
            # --------------------------------------------------------
            if eos_id is not None:
                if torch.all(next_token.squeeze(-1) == eos_id):
                    break

            # --------------------------------------------------------
            # 6. 避免超过 position embedding 长度
            # --------------------------------------------------------
            if generated.size(1) >= stage0.max_seq_len:
                break

        # 生成结束，通知 rank 1 停止。
        send_stop_signal(dst=1, device=device)

        return generated

    elif rank == 1:
        assert stage1 is not None

        step = 0

        while True:
            # --------------------------------------------------------
            # 1. 等待 rank 0 的控制信号
            # --------------------------------------------------------
            should_continue = recv_continue_or_stop(
                src=0,
                device=device,
            )

            if not should_continue:
                print("[rank 1] receive stop signal, exit loop", flush=True)
                break

            # --------------------------------------------------------
            # 2. 接收 hidden
            # --------------------------------------------------------
            hidden = recv_tensor(
                src=0,
                device=device,
                dtype=torch.float32,
            )

            print(
                f"[rank 1] step={step}, "
                f"recv hidden shape={tuple(hidden.shape)}",
                flush=True,
            )

            # --------------------------------------------------------
            # 3. Stage 1 前向
            # --------------------------------------------------------
            #
            # logits.shape = [B, current_len, vocab_size]
            logits = stage1(hidden)

            # --------------------------------------------------------
            # 4. 取最后一个位置 logits，生成 next_token
            # --------------------------------------------------------
            #
            # 自回归生成中，最后一个 token 的输出用于预测下一个 token。
            next_token_logits = logits[:, -1, :]

            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True,
            )

            print(
                f"[rank 1] step={step}, "
                f"send next_token={next_token.squeeze(-1).tolist()}",
                flush=True,
            )

            # --------------------------------------------------------
            # 5. 把 next_token 发回 rank 0
            # --------------------------------------------------------
            send_token(next_token, dst=0)

            step += 1

        return None

    else:
        raise RuntimeError("This demo only supports rank 0 and rank 1")


# ============================================================
# 7. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="使用 CPU + gloo 模拟 PP",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=5,
        help="最多生成多少个新 token",
    )

    args = parser.parse_args()

    rank, world_size, local_rank, device = init_distributed(
        use_cpu=args.cpu
    )

    # 本教学代码只实现 2-stage PP。
    assert world_size == 2, (
        "本教学版自回归 PP 代码要求 world_size=2，"
        "请使用 torchrun --nproc_per_node=2 运行。"
    )

    # ------------------------------------------------------------
    # 模型配置
    # ------------------------------------------------------------

    vocab_size = 100
    hidden_size = 128
    num_heads = 8
    intermediate_size = 512
    num_layers = 4
    max_seq_len = 64

    # 切成 2 个 stage。
    #
    # Stage 0:
    #   前 num_layers // 2 层
    #
    # Stage 1:
    #   后 num_layers - split_layer 层
    split_layer = num_layers // 2

    stage0_num_layers = split_layer
    stage1_num_layers = num_layers - split_layer

    # 为了教学可复现。
    #
    # 注意：
    #   stage0 和 stage1 本来就是不同层，不需要参数相同。
    #   真实系统中每个 stage 会从 checkpoint 加载自己负责的 layer。
    torch.manual_seed(1234)

    # ------------------------------------------------------------
    # 构建当前 rank 对应的 stage
    # ------------------------------------------------------------

    if rank == 0:
        stage0 = PipelineStage0(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            max_seq_len=max_seq_len,
            num_stage_layers=stage0_num_layers,
        ).to(device)

        stage1 = None

        print(
            f"[rank 0] Build Stage 0: "
            f"embedding + {stage0_num_layers} Transformer blocks",
            flush=True,
        )

    elif rank == 1:
        stage0 = None

        stage1 = PipelineStage1(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_heads=num_heads,
            intermediate_size=intermediate_size,
            num_stage_layers=stage1_num_layers,
        ).to(device)

        print(
            f"[rank 1] Build Stage 1: "
            f"{stage1_num_layers} Transformer blocks + final_norm + lm_head",
            flush=True,
        )

    else:
        raise RuntimeError("This demo only supports rank 0 and rank 1")

    # ------------------------------------------------------------
    # 构造 prompt
    # ------------------------------------------------------------

    if rank == 0:
        # 只有 rank 0 需要真实 token ids。
        #
        # rank 1 不接触 input_ids，它只接收 hidden states。
        prompt_ids = torch.tensor(
            [[10, 20, 30, 40]],
            dtype=torch.long,
            device=device,
        )

        print(
            f"[rank 0] prompt_ids={prompt_ids.tolist()}",
            flush=True,
        )
    else:
        prompt_ids = None

    # ------------------------------------------------------------
    # 执行自回归式 PP 生成
    # ------------------------------------------------------------

    generated = pipeline_greedy_generate_no_cache(
        rank=rank,
        device=device,
        stage0=stage0,
        stage1=stage1,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_id=None,
    )

    if rank == 0:
        print("\n========== Final Generated ==========")
        print(generated.tolist())

    cleanup_distributed()


if __name__ == "__main__":
    main()