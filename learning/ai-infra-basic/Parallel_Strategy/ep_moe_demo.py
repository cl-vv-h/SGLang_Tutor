# ep_moe_full_demo.py

"""
============================================================
Expert Parallelism / EP / MoE 完整教学版代码
============================================================

本代码目标：
    用最小但完整的分布式代码，讲清楚 MoE 模型中的 Expert Parallelism。

EP 主要用于 MoE 模型，比如：

    - Mixtral
    - DeepSeekMoE
    - Qwen-MoE
    - Switch Transformer
    - GShard

MoE 的基本思想：

    普通 Transformer FFN：

        每个 token 都经过同一个 FFN

    MoE Transformer FFN：

        有多个 Expert，每个 Expert 通常是一个 FFN。
        每个 token 先经过 Router。
        Router 选择一个或多个 Expert。
        token 只进入被选中的 Expert。

Expert Parallel 的思想：

    Router:
        每个 rank 都有完整一份，并且参数一致。

    Experts:
        不同 Expert 分布在不同 rank 上。

例如：

    world_size = 2
    num_experts = 4

    rank 0:
        Router，一份完整 Router
        Expert 0
        Expert 1

    rank 1:
        Router，一份完整 Router
        Expert 2
        Expert 3

推理时：

    1. 每个 rank 本地有一些 token hidden；
    2. 每个 rank 使用相同 Router 对本地 token 做 routing；
    3. Router 输出 global expert_id；
    4. 根据 global expert_id 找到 expert 所在 rank；
    5. 使用 all_to_all 把 token 发送到目标 expert 所在 rank；
    6. 目标 rank 用本地 expert 处理收到的 token；
    7. 再用 all_to_all 把 expert output 发回 token 原始 rank；
    8. 原始 rank 根据 token 原始位置恢复输出顺序。

重要概念：

    Router 是复制的：
        每个 rank 都有相同 Router 参数。

    Expert 是分片的：
        每个 rank 只持有一部分 expert。

    Routing 是全局的：
        Router 不是只在本地 experts 里选，而是在所有 experts 里选。

    通信模式：
        EP 最典型的通信是 all_to_all。

注意：

    这是一份教学代码，不是工业高性能 MoE 实现。

    工业实现还会涉及：
        - Top-2 / Top-k routing
        - capacity factor
        - token dropping
        - grouped GEMM
        - fused dispatch/combine kernel
        - variable-size all-to-all
        - expert 内部 TP
        - EP + TP + PP + DP 混合并行
        - load balancing
        - router jitter noise
        - shared experts
"""


import os
import argparse
from typing import Tuple, Dict, List

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

        torchrun --nproc_per_node=2 ep_moe_full_demo.py

    torchrun 会自动设置几个环境变量：

        RANK:
            当前进程在全局分布式任务中的编号。

        WORLD_SIZE:
            当前分布式任务一共有多少个进程。

        LOCAL_RANK:
            当前进程在本机上的编号，通常用于绑定 GPU。

    在本代码中：

        world_size = EP size

    也就是说：

        experts 会被分布到 world_size 个 rank 上。

    举例：

        world_size = 2
        num_experts = 4

        每个 rank 持有 2 个 experts。
    """

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    if use_cpu:
        # CPU 模拟时使用 gloo。
        backend = "gloo"
        device = torch.device("cpu")
    else:
        # GPU 场景一般使用 nccl。
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
    销毁分布式通信组。
    """

    dist.destroy_process_group()


# ============================================================
# 2. 同步 Router 参数
# ============================================================

def broadcast_module_parameters(
    module: nn.Module,
    src: int = 0,
):
    """
    将某个模块的参数从 src rank 广播到所有 rank。

    为什么需要这个函数？

        在 EP 中，Router 通常在每个 rank 上都有一份完整副本。
        但是这些 Router 副本必须参数一致。

        如果 Router 不一致，那么同一个 token hidden 在不同 rank 上
        可能会被路由到不同 expert，这会破坏模型语义。

    真实系统中：

        一般从 checkpoint 加载相同 Router 参数。
        这里为了教学，使用 broadcast 确保一致。

    注意：

        这个函数会遍历 module 的所有参数和 buffer。
    """

    for param in module.parameters():
        dist.broadcast(param.data, src=src)

    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=src)


# ============================================================
# 3. Top-1 Router
# ============================================================

class Top1Router(nn.Module):
    """
    Top-1 Router。

    Router 的作用：

        对每个 token hidden 计算它应该进入哪个 Expert。

    输入：

        x.shape = [T, H]

        T:
            当前 rank 本地 token 数。

        H:
            hidden_size。

    输出：

        expert_ids.shape = [T]

            每个 token 被路由到的 global expert id。

        gate_values.shape = [T, 1]

            每个 token 对应被选中 expert 的概率值。

    Router 本质上是一个线性分类器：

        router_logits = x @ W_router

        router_logits.shape = [T, num_experts]

    其中：

        num_experts 是全局 expert 数量，不是本地 expert 数量。

    也就是说：

        Router 是全局路由。

    例如：

        num_experts = 4

        对某个 token，Router 可能输出：

            expert_id = 3

        即使当前 rank 只持有 expert 0/1，这个 token 也可以被路由到 expert 3，
        然后通过 all_to_all 被发送到 expert 3 所在 rank。
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_experts = num_experts

        # Router 是一个 Linear：
        #
        #   hidden_size -> num_experts
        #
        # 对每个 token 输出所有 expert 的分数。
        self.router = nn.Linear(
            hidden_size,
            num_experts,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行 Top-1 routing。

        参数：

            x.shape = [T, H]

        返回：

            expert_ids:
                shape = [T]
                每个 token 选择的 global expert id。

            gate_values:
                shape = [T, 1]
                每个 token 对应被选 expert 的概率。

            router_probs:
                shape = [T, num_experts]
                每个 token 对所有 expert 的概率分布。
                这里只是为了 debug 方便返回。
        """

        # router_logits.shape = [T, num_experts]
        router_logits = self.router(x)

        # 对 expert 维度做 softmax，得到选择每个 expert 的概率。
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-1 routing：
        #
        # 对每个 token，选择概率最大的 expert。
        #
        # gate_values.shape = [T]
        # expert_ids.shape = [T]
        gate_values, expert_ids = torch.max(
            router_probs,
            dim=-1,
        )

        # 变成 [T, 1]，方便后面和 expert output 相乘。
        gate_values = gate_values.unsqueeze(-1)

        return expert_ids, gate_values, router_probs


# ============================================================
# 4. Expert FFN
# ============================================================

class Expert(nn.Module):
    """
    一个 Expert。

    在 MoE Transformer 中，Expert 通常就是一个 FFN。

    普通 FFN：

        hidden_size -> intermediate_size -> hidden_size

    一个 MoE 层中有很多个 Experts。

    每个 token 只进入其中一个或几个 Experts。

    在本代码中，我们使用 Top-1 routing：
        每个 token 只进入一个 Expert。
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        expert_id: int,
    ):
        super().__init__()

        self.expert_id = expert_id

        self.up_proj = nn.Linear(
            hidden_size,
            intermediate_size,
        )

        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x.shape = [N, H]

        N:
            当前 Expert 收到的 token 数量。

        返回：
            out.shape = [N, H]
        """

        x = self.up_proj(x)
        x = F.gelu(x)
        x = self.down_proj(x)

        return x


# ============================================================
# 5. Expert Parallel MoE Layer
# ============================================================

class ExpertParallelMoE(nn.Module):
    """
    Expert Parallel MoE Layer。

    这是本代码最核心的类。

    输入：

        x.shape = [T, H]

        表示当前 rank 本地有 T 个 token hidden。

    输出：

        out.shape = [T, H]

        表示当前 rank 的这 T 个 token 经过 MoE 层后的输出。
        输出顺序必须和输入 x 的 token 顺序一致。

    关键流程：

        1. Router routing
            当前 rank 使用复制的 Router 对本地 tokens 做全局 expert 路由。

        2. Build dispatch buffer
            把本地 tokens 按目标 expert 所在 rank 分桶。

        3. all_to_all dispatch
            把 token 发送到 expert 所在 rank。

        4. Local expert compute
            当前 rank 用本地 experts 处理收到的 tokens。

        5. all_to_all combine
            把 expert 输出发送回 token 原始 rank。

        6. Restore order
            原始 rank 按 token 原始位置恢复输出顺序。

    假设：

        num_experts = 4
        world_size = 2

    则：

        experts_per_rank = 2

        rank 0:
            Expert 0
            Expert 1

        rank 1:
            Expert 2
            Expert 3
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        rank: int,
        world_size: int,
        capacity: int,
    ):
        super().__init__()

        assert num_experts % world_size == 0, (
            "教学版要求 num_experts 能被 world_size 整除"
        )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.rank = rank
        self.world_size = world_size
        self.capacity = capacity

        # 每个 rank 持有多少 expert。
        self.experts_per_rank = num_experts // world_size

        # 当前 rank 持有的 global expert id 范围。
        self.local_expert_start = rank * self.experts_per_rank
        self.local_expert_end = self.local_expert_start + self.experts_per_rank

        # ------------------------------------------------------------
        # Router：每个 rank 都有完整一份
        # ------------------------------------------------------------
        #
        # 重要：
        #   这个 Router 的输出是 global expert id。
        #
        #   所有 rank 上的 Router 参数应该一致。
        #   本代码会在模型构建后 broadcast Router 参数，确保一致。
        self.router = Top1Router(
            hidden_size=hidden_size,
            num_experts=num_experts,
        )

        # ------------------------------------------------------------
        # Experts：每个 rank 只创建本 rank 负责的 experts
        # ------------------------------------------------------------
        self.local_experts = nn.ModuleDict()

        for expert_id in range(
            self.local_expert_start,
            self.local_expert_end,
        ):
            self.local_experts[str(expert_id)] = Expert(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                expert_id=expert_id,
            )

    # ------------------------------------------------------------
    # expert_id -> rank 映射
    # ------------------------------------------------------------

    def expert_to_rank(self, expert_id: int) -> int:
        """
        根据 global expert id 找到该 expert 所在的 rank。

        假设：

            num_experts = 4
            world_size = 2
            experts_per_rank = 2

        则：

            expert 0 -> rank 0
            expert 1 -> rank 0
            expert 2 -> rank 1
            expert 3 -> rank 1

        公式：

            expert_rank = expert_id // experts_per_rank
        """

        return expert_id // self.experts_per_rank

    def is_local_expert(self, expert_id: int) -> bool:
        """
        判断某个 expert_id 是否属于当前 rank。
        """

        return self.local_expert_start <= expert_id < self.local_expert_end

    # ------------------------------------------------------------
    # Debug 工具
    # ------------------------------------------------------------

    def print_local_experts(self):
        """
        打印当前 rank 持有的 expert。
        """

        expert_ids = list(range(self.local_expert_start, self.local_expert_end))

        print(
            f"[rank {self.rank}] local experts = {expert_ids}",
            flush=True,
        )

    # ------------------------------------------------------------
    # Dispatch buffer 构建
    # ------------------------------------------------------------

    def build_dispatch_buffers(
        self,
        x: torch.Tensor,
        expert_ids: torch.Tensor,
        gate_values: torch.Tensor,
    ):
        """
        构造 dispatch 阶段的 send buffers。

        输入：

            x.shape = [T, H]

                当前 rank 本地 token hidden。

            expert_ids.shape = [T]

                每个 token 选择的 global expert id。

            gate_values.shape = [T, 1]

                每个 token 对应 expert 的 router probability。

        输出：

            send_tokens.shape = [world_size, capacity, H]

                send_tokens[dst_rank, slot] 表示：
                    当前 rank 要发送给 dst_rank 的第 slot 个 token hidden。

            send_indices.shape = [world_size, capacity]

                send_indices[dst_rank, slot] 表示：
                    这个 token 在当前 rank 原始输入 x 中的 token_idx。

                为什么要记录？
                    因为 expert 输出最终要发回当前 rank，
                    当前 rank 需要知道这个输出对应 out[token_idx]。

            send_expert_ids.shape = [world_size, capacity]

                send_expert_ids[dst_rank, slot] 表示：
                    这个 token 要进入哪个 global expert。

            send_gates.shape = [world_size, capacity, 1]

                send_gates[dst_rank, slot] 表示：
                    这个 token 的 gate value。

            send_mask.shape = [world_size, capacity]

                True:
                    该槽位是真实 token。

                False:
                    该槽位是 padding。

        为什么要 capacity？

            all_to_all_single 最容易使用的是固定大小 tensor。
            但是每个 rank 发往不同目标 rank 的 token 数量可能不同。

            所以教学代码里给每个 dst_rank 预留 capacity 个槽位。

        举例：

            当前 rank 有 4 个 token：

                token0 -> expert 3 -> dst_rank 1
                token1 -> expert 0 -> dst_rank 0
                token2 -> expert 2 -> dst_rank 1
                token3 -> expert 1 -> dst_rank 0

            那么：

                send_tokens[0] 里放 token1, token3
                send_tokens[1] 里放 token0, token2
        """

        T, H = x.shape
        device = x.device

        # 发送 token hidden。
        send_tokens = torch.zeros(
            self.world_size,
            self.capacity,
            H,
            dtype=x.dtype,
            device=device,
        )

        # 发送 token 原始位置。
        send_indices = torch.full(
            (self.world_size, self.capacity),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )

        # 发送 token 对应 expert id。
        send_expert_ids = torch.full(
            (self.world_size, self.capacity),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )

        # 发送 gate value。
        send_gates = torch.zeros(
            self.world_size,
            self.capacity,
            1,
            dtype=x.dtype,
            device=device,
        )

        # 有效槽位 mask。
        send_mask = torch.zeros(
            self.world_size,
            self.capacity,
            dtype=torch.bool,
            device=device,
        )

        # offsets[dst_rank] 记录当前已经给 dst_rank 放了多少 token。
        offsets = [0 for _ in range(self.world_size)]

        for token_idx in range(T):
            # 当前 token 的 global expert id。
            expert_id = int(expert_ids[token_idx].item())

            # 该 expert 所在的 rank。
            dst_rank = self.expert_to_rank(expert_id)

            # 放到 send_tokens[dst_rank] 的哪个 slot。
            slot = offsets[dst_rank]

            if slot >= self.capacity:
                raise RuntimeError(
                    f"[rank {self.rank}] capacity overflow: "
                    f"dst_rank={dst_rank}, capacity={self.capacity}. "
                    f"可以增大 capacity，或者实现 token dropping。"
                )

            # 把 token hidden 放到目标 rank 对应的 bucket 里。
            send_tokens[dst_rank, slot] = x[token_idx]

            # 记录这个 token 在当前 rank 原始输入中的位置。
            send_indices[dst_rank, slot] = token_idx

            # 记录它要进入哪个 global expert。
            send_expert_ids[dst_rank, slot] = expert_id

            # 记录 gate value。
            send_gates[dst_rank, slot] = gate_values[token_idx]

            # 标记该 slot 有效。
            send_mask[dst_rank, slot] = True

            offsets[dst_rank] += 1

        # Debug：打印当前 rank 发给每个目标 rank 的 token 数量。
        counts = [int(send_mask[r].sum().item()) for r in range(self.world_size)]

        print(
            f"[rank {self.rank}] dispatch send counts per dst rank = {counts}",
            flush=True,
        )

        return (
            send_tokens,
            send_indices,
            send_expert_ids,
            send_gates,
            send_mask,
        )

    # ------------------------------------------------------------
    # all_to_all
    # ------------------------------------------------------------

    def all_to_all_tensor(self, send_tensor: torch.Tensor) -> torch.Tensor:
        """
        对 send_tensor 执行 all_to_all_single。

        输入：

            send_tensor.shape = [world_size, capacity, ...]

        语义：

            send_tensor[0] 发送给 rank 0
            send_tensor[1] 发送给 rank 1
            ...
            send_tensor[i] 发送给 rank i

        返回：

            recv_tensor.shape = [world_size, capacity, ...]

        语义：

            recv_tensor[0] 是从 rank 0 收到的数据
            recv_tensor[1] 是从 rank 1 收到的数据
            ...
            recv_tensor[i] 是从 rank i 收到的数据

        举例：

            rank 0 上：

                send_tensor[1] = rank 0 要发给 rank 1 的 token

            rank 1 上 all_to_all 后：

                recv_tensor[0] = rank 1 从 rank 0 收到的 token

        为什么 EP 用 all_to_all？

            因为每个 rank 的 token 都可能被路由到任意 expert。
            每个 expert 又可能在任意 rank 上。

            所以通信模式是：
                every rank sends to every rank
        """

        recv_tensor = torch.empty_like(send_tensor)

        dist.all_to_all_single(
            output=recv_tensor,
            input=send_tensor,
        )

        return recv_tensor

    # ------------------------------------------------------------
    # 本地 expert 计算
    # ------------------------------------------------------------

    def compute_local_experts(
        self,
        recv_tokens: torch.Tensor,
        recv_expert_ids: torch.Tensor,
        recv_gates: torch.Tensor,
        recv_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        当前 rank 对收到的 tokens 执行本地 experts。

        输入：

            recv_tokens.shape = [world_size, capacity, H]

                recv_tokens[src_rank, slot]
                    表示从 src_rank 发到当前 rank 的 token hidden。

            recv_expert_ids.shape = [world_size, capacity]

                表示这个 token 要进入哪个 global expert。

            recv_gates.shape = [world_size, capacity, 1]

                表示 token 的 gate value。

            recv_mask.shape = [world_size, capacity]

                表示槽位是否有效。

        输出：

            processed.shape = [world_size, capacity, H]

                processed[src_rank, slot]
                    表示从 src_rank 发来的 token
                    经过当前 rank 的本地 expert 后的输出。

        注意：

            当前 rank 只持有一部分 experts。

            因此这里只有 expert_id 属于当前 rank 的 token 才会被处理。

        由于 dispatch 阶段已经按照 expert_id 发到目标 rank，
        理论上当前 rank 收到的有效 token 都应该属于当前 rank 的 experts。
        """

        processed = torch.zeros_like(recv_tokens)

        # 当前 rank 收到的有效 token 数量。
        total_received = int(recv_mask.sum().item())

        print(
            f"[rank {self.rank}] total received tokens = {total_received}",
            flush=True,
        )

        # 遍历当前 rank 持有的所有 experts。
        for expert_id in range(
            self.local_expert_start,
            self.local_expert_end,
        ):
            # 找出所有路由到这个 expert 的 token。
            #
            # token_mask.shape = [world_size, capacity]
            token_mask = (recv_expert_ids == expert_id) & recv_mask

            num_tokens_for_expert = int(token_mask.sum().item())

            print(
                f"[rank {self.rank}] expert {expert_id} receives "
                f"{num_tokens_for_expert} tokens",
                flush=True,
            )

            if num_tokens_for_expert == 0:
                continue

            # 取出这些 token hidden。
            #
            # selected_tokens.shape = [N, H]
            selected_tokens = recv_tokens[token_mask]

            # 找到本地 expert。
            expert = self.local_experts[str(expert_id)]

            # 执行 Expert FFN。
            #
            # expert_out.shape = [N, H]
            expert_out = expert(selected_tokens)

            # Top-1 MoE 中，通常会乘上 gate value。
            #
            # selected_gates.shape = [N, 1]
            selected_gates = recv_gates[token_mask]

            expert_out = expert_out * selected_gates

            # 写回 processed 的对应位置。
            processed[token_mask] = expert_out

        return processed

    # ------------------------------------------------------------
    # 恢复输出顺序
    # ------------------------------------------------------------

    def restore_output_order(
        self,
        returned_outputs: torch.Tensor,
        returned_indices: torch.Tensor,
        returned_mask: torch.Tensor,
        num_local_tokens: int,
    ) -> torch.Tensor:
        """
        将 combine 回来的 expert outputs 恢复到当前 rank 的原 token 顺序。

        输入：

            returned_outputs.shape = [world_size, capacity, H]

                returned_outputs[src, slot]
                    表示某个 expert rank 返回给当前 rank 的输出。

                注意：
                    这里第 0 维 src 表示“这个结果来自哪个 expert rank”。

            returned_indices.shape = [world_size, capacity]

                returned_indices[src, slot]
                    表示这个输出对应当前 rank 原始输入 x 的哪个 token_idx。

            returned_mask.shape = [world_size, capacity]

                表示该槽位是否有效。

            num_local_tokens:
                当前 rank 原始输入 token 数 T。

        输出：

            out.shape = [T, H]

                out[token_idx] 是原始 token_idx 对应的 MoE 输出。
        """

        H = returned_outputs.size(-1)

        out = torch.zeros(
            num_local_tokens,
            H,
            dtype=returned_outputs.dtype,
            device=returned_outputs.device,
        )

        # 遍历所有返回结果。
        for src_rank in range(self.world_size):
            for slot in range(self.capacity):
                if not bool(returned_mask[src_rank, slot].item()):
                    continue

                token_idx = int(returned_indices[src_rank, slot].item())

                if token_idx < 0:
                    continue

                out[token_idx] = returned_outputs[src_rank, slot]

        return out

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ExpertParallelMoE 前向过程。

        输入：

            x.shape = [T, H]

        输出：

            out.shape = [T, H]

        详细流程：

            Step 1:
                Router 对本地 token 做全局 expert 选择。

            Step 2:
                构造 dispatch buffers。

            Step 3:
                all_to_all dispatch，把 token 发送到 expert 所在 rank。

            Step 4:
                当前 rank 执行本地 experts。

            Step 5:
                all_to_all combine，把 expert output 发送回 token 原始 rank。

            Step 6:
                恢复输出顺序。
        """

        T, H = x.shape

        # ============================================================
        # Step 1. Router：本地 token -> global expert_id
        # ============================================================

        expert_ids, gate_values, router_probs = self.router(x)

        print(
            f"[rank {self.rank}] router expert_ids = {expert_ids.tolist()}",
            flush=True,
        )

        # ============================================================
        # Step 2. 构造 dispatch buffers
        # ============================================================

        (
            send_tokens,
            send_indices,
            send_expert_ids,
            send_gates,
            send_mask,
        ) = self.build_dispatch_buffers(
            x=x,
            expert_ids=expert_ids,
            gate_values=gate_values,
        )

        # ============================================================
        # Step 3. all_to_all dispatch
        # ============================================================
        #
        # 下面这些 tensor 都要一起 all_to_all。
        #
        # tokens:
        #   真正要给 expert 处理的 hidden。
        #
        # indices:
        #   token 原始位置，用于 combine 后恢复顺序。
        #
        # expert_ids:
        #   告诉目标 rank，这个 token 应该进入哪个 expert。
        #
        # gates:
        #   gate value，用于加权 expert output。
        #
        # mask:
        #   标记有效槽位。
        recv_tokens = self.all_to_all_tensor(send_tokens)
        recv_indices = self.all_to_all_tensor(send_indices)
        recv_expert_ids = self.all_to_all_tensor(send_expert_ids)
        recv_gates = self.all_to_all_tensor(send_gates)
        recv_mask = self.all_to_all_tensor(send_mask)

        # ============================================================
        # Step 4. 当前 rank 执行本地 experts
        # ============================================================

        processed = self.compute_local_experts(
            recv_tokens=recv_tokens,
            recv_expert_ids=recv_expert_ids,
            recv_gates=recv_gates,
            recv_mask=recv_mask,
        )

        # ============================================================
        # Step 5. all_to_all combine
        # ============================================================
        #
        # dispatch 时：
        #   原始 rank -> expert rank
        #
        # combine 时：
        #   expert rank -> 原始 rank
        #
        # processed[src_rank] 应该发回 src_rank。
        #
        # recv_indices/recv_mask 也要一起发回去，
        # 因为原始 rank 需要知道：
        #   这个 output 对应哪个 token_idx。
        returned_outputs = self.all_to_all_tensor(processed)
        returned_indices = self.all_to_all_tensor(recv_indices)
        returned_mask = self.all_to_all_tensor(recv_mask)

        # ============================================================
        # Step 6. 恢复输出顺序
        # ============================================================

        out = self.restore_output_order(
            returned_outputs=returned_outputs,
            returned_indices=returned_indices,
            returned_mask=returned_mask,
            num_local_tokens=T,
        )

        return out


# ============================================================
# 6. 一个包含 MoE 层的小模型
# ============================================================

class TinyMoEModel(nn.Module):
    """
    一个极简 MoE 模型。

    这个模型不是完整 Transformer，只是为了聚焦 EP 机制。

    结构：

        input_ids
            ↓
        embedding
            ↓
        dense projection
            ↓
        ExpertParallelMoE
            ↓
        residual add
            ↓
        norm
            ↓
        lm_head
            ↓
        logits

    在真实 MoE Transformer 中，结构一般是：

        Attention
        MoE FFN
        Attention
        MoE FFN
        ...

    也就是说 MoE 通常替代的是 Transformer Block 里的 FFN。
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        rank: int,
        world_size: int,
        capacity: int,
    ):
        super().__init__()

        self.rank = rank
        self.world_size = world_size

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_size,
        )

        self.dense = nn.Linear(
            hidden_size,
            hidden_size,
        )

        self.moe = ExpertParallelMoE(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_experts=num_experts,
            rank=rank,
            world_size=world_size,
            capacity=capacity,
        )

        self.norm = nn.LayerNorm(hidden_size)

        self.lm_head = nn.Linear(
            hidden_size,
            vocab_size,
            bias=False,
        )

    def sync_shared_parameters(self):
        """
        同步共享模块的参数。

        在 EP 中：

            Router 是共享模块，所有 rank 应该一致。

        在这份教学模型里，embedding/dense/norm/lm_head 也每个 rank 都有一份，
        如果要严格保证模型副本一致，也可以同步它们。

        但是 EP 语义上最关键的是 Router 一致。

        这里为了严谨，统一同步以下共享模块：

            embedding
            dense
            moe.router
            norm
            lm_head

        local experts 不同步，因为每个 rank 持有的是不同 experts。
        """

        broadcast_module_parameters(self.embedding, src=0)
        broadcast_module_parameters(self.dense, src=0)
        broadcast_module_parameters(self.moe.router, src=0)
        broadcast_module_parameters(self.norm, src=0)
        broadcast_module_parameters(self.lm_head, src=0)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids.shape = [T]

        表示当前 rank 本地有 T 个 token。

        输出：

            logits.shape = [T, vocab_size]
        """

        # [T] -> [T, H]
        x = self.embedding(input_ids)

        # 一个普通 dense 层，模拟 MoE 前的 hidden transformation。
        x = self.dense(x)

        # 保存 residual。
        residual = x

        # MoE 层。
        moe_out = self.moe(x)

        # 残差连接。
        x = residual + moe_out

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits


# ============================================================
# 7. 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="使用 CPU + gloo 后端模拟 EP",
    )

    args = parser.parse_args()

    rank, world_size, local_rank, device = init_distributed(args.cpu)

    # ------------------------------------------------------------
    # 配置
    # ------------------------------------------------------------

    vocab_size = 100
    hidden_size = 64
    intermediate_size = 128

    # 全局 expert 数。
    #
    # 例如 num_experts = 4，world_size = 2：
    #   rank 0 持有 expert 0/1
    #   rank 1 持有 expert 2/3
    num_experts = 4

    assert num_experts % world_size == 0, (
        "num_experts 必须能被 world_size 整除"
    )

    # 每个 rank 本地 token 数。
    local_num_tokens = 6

    # capacity 表示：
    #
    #   当前 rank 最多可以向某个目标 rank 发送多少 token。
    #
    # 教学代码中设置成 local_num_tokens，足够容纳最坏情况：
    #   当前 rank 的所有 token 都发往同一个目标 rank。
    #
    # 真实 MoE 中 capacity 通常按 capacity_factor 计算。
    capacity = local_num_tokens

    # ------------------------------------------------------------
    # 初始化模型
    # ------------------------------------------------------------
    #
    # 注意：
    #   这里先使用不同 seed 初始化整个模型。
    #   然后通过 sync_shared_parameters 同步共享模块。
    #
    #   local experts 本来就是分片的，不需要在不同 rank 同步。
    torch.manual_seed(1000 + rank)

    model = TinyMoEModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        rank=rank,
        world_size=world_size,
        capacity=capacity,
    ).to(device)

    # 确保 Router 和其他共享模块参数一致。
    model.sync_shared_parameters()

    model.eval()

    # 打印当前 rank 持有的 experts。
    model.moe.print_local_experts()

    # ------------------------------------------------------------
    # 构造当前 rank 本地输入 token
    # ------------------------------------------------------------
    #
    # 模拟当前 rank 本地有一批 token hidden。
    #
    # rank 0:
    #   token ids: 0,1,2,3,4,5
    #
    # rank 1:
    #   token ids: 10,11,12,13,14,15
    #
    # 注意：
    #   这些 token 只是当前 rank 本地 token。
    #   它们经过 Router 后可以被发送到任意 rank 的 expert。
    local_input_ids = torch.arange(
        rank * 10,
        rank * 10 + local_num_tokens,
        dtype=torch.long,
        device=device,
    )

    print(
        f"[rank {rank}] local_input_ids = {local_input_ids.tolist()}",
        flush=True,
    )

    # ------------------------------------------------------------
    # 执行前向
    # ------------------------------------------------------------

    with torch.no_grad():
        logits = model(local_input_ids)

    print(
        f"[rank {rank}] logits.shape = {tuple(logits.shape)}",
        flush=True,
    )

    cleanup_distributed()


if __name__ == "__main__":
    main()