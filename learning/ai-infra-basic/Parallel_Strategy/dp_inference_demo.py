# dp_inference_demo.py

"""
教学目标：
    用最小代码理解“大模型推理中的数据并行 DP”。

重点：
    1. 每个 worker 拥有一份完整模型；
    2. 主进程负责把请求分发给不同 worker；
    3. worker 之间不需要做 attention 通信、参数通信、KV Cache 通信；
    4. 每个请求只在一个 worker 上完整执行；
    5. DP 提升的是多请求吞吐，不降低单请求延迟，也不能解决单卡放不下模型的问题。

注意：
    这是教学版代码，不是高性能 serving 框架。
    真实 SGLang/vLLM 中会有更复杂的 scheduler、tokenizer、KV Cache manager、batching、streaming response 等。
"""

import os
import time
import queue
import random
import multiprocessing as mp
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 定义一个极简 Causal LM 模型
# ============================================================

class TinyCausalLM(nn.Module):
    """
    一个非常小的 decoder-only 语言模型。

    这里不是为了实现真正高质量生成，
    而是为了模拟“大模型推理”的接口：

        input_ids -> logits -> next_token

    真实大模型中这里会是：
        Embedding
        Transformer Block × N
        KV Cache
        lm_head

    这里为了让 DP 逻辑更清晰，用一个小模型代替。
    """

    def __init__(
        self,
        vocab_size: int = 100,
        d_model: int = 128,
        max_seq_len: int = 128,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # token embedding:
        #   input_ids.shape = [batch_size, seq_len]
        #   token_emb.shape = [batch_size, seq_len, d_model]
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # position embedding:
        #   用于表示 token 在序列中的位置。
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # 一个简单的 MLP，用来模拟 Transformer block 的计算。
        #
        # 真实模型这里会是多层 self-attention + FFN。
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # 输出层:
        #   hidden -> vocab logits
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            input_ids:
                shape = [batch_size, seq_len]

        返回:
            logits:
                shape = [batch_size, seq_len, vocab_size]

        推理时通常只关心最后一个位置:
            logits[:, -1, :]
        """

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 构造 position_ids:
        #
        # 如果 seq_len = 5:
        #   position_ids = [0, 1, 2, 3, 4]
        #
        # 再扩展到 batch:
        #   [batch_size, seq_len]
        position_ids = torch.arange(
            seq_len,
            device=device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch_size, seq_len)

        # token embedding
        token_emb = self.token_embedding(input_ids)

        # position embedding
        pos_emb = self.position_embedding(position_ids)

        # token embedding + position embedding
        hidden = token_emb + pos_emb

        # 模拟 Transformer block
        hidden = hidden + self.ffn(hidden)

        # 映射到词表
        logits = self.lm_head(hidden)

        return logits


# ============================================================
# 2. 定义请求和响应结构
# ============================================================

@dataclass
class InferenceRequest:
    """
    一个推理请求。

    request_id:
        请求编号，用于主进程收集结果时区分不同请求。

    prompt_ids:
        输入 token id。

    max_new_tokens:
        最多生成多少个新 token。
    """

    request_id: int
    prompt_ids: List[int]
    max_new_tokens: int


@dataclass
class InferenceResponse:
    """
    一个推理响应。

    request_id:
        对应请求编号。

    worker_id:
        由哪个 worker 处理。

    generated_ids:
        生成后的完整 token 序列。
    """

    request_id: int
    worker_id: int
    generated_ids: List[int]


# ============================================================
# 3. 单请求 greedy generate
# ============================================================

@torch.no_grad()
def greedy_generate(
    model: TinyCausalLM,
    prompt_ids: List[int],
    max_new_tokens: int,
    device: torch.device,
) -> List[int]:
    """
    对单个请求做贪心生成。

    这里为了突出 DP，不引入复杂 sampling。
    每一步直接选择 logits 最大的 token。

    参数:
        model:
            当前 worker 上的完整模型副本。

        prompt_ids:
            输入 prompt token id list。

        max_new_tokens:
            最多生成多少个新 token。

        device:
            当前 worker 使用的设备。

    返回:
        generated:
            prompt + 新生成 token。
    """

    # generated 是当前已经生成的完整序列。
    #
    # 推理开始时，它等于 prompt。
    generated = list(prompt_ids)

    for _ in range(max_new_tokens):
        # 把 Python list 转成 Tensor。
        #
        # shape = [1, current_seq_len]
        input_ids = torch.tensor(
            [generated],
            dtype=torch.long,
            device=device,
        )

        # 模型前向。
        #
        # logits.shape = [1, current_seq_len, vocab_size]
        logits = model(input_ids)

        # 只取最后一个位置的 logits。
        #
        # 因为自回归生成中，最后一个位置用于预测下一个 token。
        #
        # next_token_logits.shape = [vocab_size]
        next_token_logits = logits[0, -1, :]

        # greedy decoding:
        #   选择分数最高的 token。
        next_token = int(torch.argmax(next_token_logits).item())

        # 把新 token 追加到序列后面。
        generated.append(next_token)

        # 为了防止超过 position embedding 最大长度。
        #
        # 真实大模型也会有 max context length 限制。
        if len(generated) >= model.max_seq_len:
            break

    return generated


# ============================================================
# 4. Worker 进程函数
# ============================================================

def worker_loop(
    worker_id: int,
    num_workers: int,
    request_queue: mp.Queue,
    response_queue: mp.Queue,
    vocab_size: int,
    d_model: int,
    max_seq_len: int,
):
    """
    每个 worker 进程执行这个函数。

    一个 worker 对应一个模型副本。

    在真实推理 DP 中：
        worker 0 通常绑定 GPU 0；
        worker 1 通常绑定 GPU 1；
        worker 2 通常绑定 GPU 2。

    每个 worker 内部有完整模型参数。

    这个函数做的事情：
        1. 选择当前 worker 对应的 device；
        2. 初始化完整模型；
        3. 不断从 request_queue 中取请求；
        4. 对请求做推理；
        5. 把结果写入 response_queue。
    """

    # ------------------------------------------------------------
    # 1. 绑定设备
    # ------------------------------------------------------------

    if torch.cuda.is_available():
        # 如果有 GPU，则每个 worker 使用一张 GPU。
        #
        # worker_id = 0 -> cuda:0
        # worker_id = 1 -> cuda:1
        #
        # 如果 GPU 数量少于 worker 数量，可以用取模。
        num_gpus = torch.cuda.device_count()
        device_id = worker_id % num_gpus
        device = torch.device(f"cuda:{device_id}")

        # 设置当前进程默认 CUDA device。
        torch.cuda.set_device(device)
    else:
        # 没有 GPU 时，用 CPU 模拟多 worker。
        device = torch.device("cpu")

    print(
        f"[Worker {worker_id}] start on device={device}, pid={os.getpid()}",
        flush=True,
    )

    # ------------------------------------------------------------
    # 2. 每个 worker 加载一份完整模型
    # ------------------------------------------------------------

    model = TinyCausalLM(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=max_seq_len,
    ).to(device)

    model.eval()

    # 注意：
    #   这里每个 worker 初始化的是一份独立模型。
    #
    # 真实 serving 中：
    #   每个 worker 会从同一个 checkpoint 加载相同参数。
    #
    # 为了 demo 简单，这里没有保存/加载 checkpoint。
    # 如果要保证每个 worker 参数完全一致，可以设置相同随机种子或加载相同 state_dict。

    # ------------------------------------------------------------
    # 3. 循环处理请求
    # ------------------------------------------------------------

    while True:
        # 从队列中取请求。
        #
        # 主进程会把 InferenceRequest 放进来。
        # 如果收到 None，表示退出。
        req = request_queue.get()

        if req is None:
            print(f"[Worker {worker_id}] received stop signal", flush=True)
            break

        assert isinstance(req, InferenceRequest)

        print(
            f"[Worker {worker_id}] processing request {req.request_id}, "
            f"prompt_len={len(req.prompt_ids)}",
            flush=True,
        )

        # --------------------------------------------------------
        # 4. 执行推理
        # --------------------------------------------------------

        generated_ids = greedy_generate(
            model=model,
            prompt_ids=req.prompt_ids,
            max_new_tokens=req.max_new_tokens,
            device=device,
        )

        # --------------------------------------------------------
        # 5. 返回结果
        # --------------------------------------------------------

        resp = InferenceResponse(
            request_id=req.request_id,
            worker_id=worker_id,
            generated_ids=generated_ids,
        )

        response_queue.put(resp)

    print(f"[Worker {worker_id}] exit", flush=True)


# ============================================================
# 5. Dispatcher / Router
# ============================================================

class RoundRobinDispatcher:
    """
    一个最简单的 round-robin 请求分发器。

    它的作用类似真实推理系统里的 router / load balancer。

    对于第 i 个请求：
        分发给 worker_id = i % num_workers

    真实系统中分发策略可能更复杂，例如：
        1. 按 worker 当前负载；
        2. 按队列长度；
        3. 按 prompt 长度；
        4. 按 KV Cache 命中情况；
        5. 按租户或优先级；
        6. 按模型副本健康状态。
    """

    def __init__(self, num_workers: int):
        self.num_workers = num_workers
        self.next_worker = 0

    def dispatch(self, request: InferenceRequest) -> int:
        """
        返回应该处理该请求的 worker_id。
        """

        worker_id = self.next_worker

        self.next_worker = (self.next_worker + 1) % self.num_workers

        return worker_id


# ============================================================
# 6. 主进程：创建 worker，分发请求，收集结果
# ============================================================

def main():
    """
    主流程。

    这就是一个最小 DP 推理系统：

        1. 创建多个 worker；
        2. 每个 worker 加载完整模型；
        3. 主进程构造多个请求；
        4. dispatcher 将请求分发给不同 worker；
        5. worker 并行处理请求；
        6. 主进程收集响应。
    """

    # ------------------------------------------------------------
    # 1. 配置参数
    # ------------------------------------------------------------

    vocab_size = 100
    d_model = 128
    max_seq_len = 64

    # 如果有 GPU，则默认使用 GPU 数量作为 worker 数。
    # 如果没有 GPU，就用 2 个 CPU worker 模拟。
    if torch.cuda.is_available():
        num_workers = torch.cuda.device_count()
    else:
        num_workers = 2

    print(f"num_workers = {num_workers}")

    # ------------------------------------------------------------
    # 2. 创建队列
    # ------------------------------------------------------------

    # 每个 worker 一个 request queue。
    #
    # 这样主进程可以精确地把请求发给某个 worker。
    request_queues = [
        mp.Queue()
        for _ in range(num_workers)
    ]

    # 所有 worker 共用一个 response queue。
    #
    # worker 推理完成后，把结果放到这个队列里。
    response_queue = mp.Queue()

    # ------------------------------------------------------------
    # 3. 启动 worker 进程
    # ------------------------------------------------------------

    workers = []

    for worker_id in range(num_workers):
        p = mp.Process(
            target=worker_loop,
            args=(
                worker_id,
                num_workers,
                request_queues[worker_id],
                response_queue,
                vocab_size,
                d_model,
                max_seq_len,
            ),
        )

        p.start()
        workers.append(p)

    # ------------------------------------------------------------
    # 4. 构造一些模拟请求
    # ------------------------------------------------------------

    random.seed(0)

    requests: List[InferenceRequest] = []

    for request_id in range(8):
        # 随机 prompt 长度。
        prompt_len = random.randint(4, 10)

        # token id 随机生成。
        #
        # 为了简单，token id 范围是 [1, vocab_size)。
        prompt_ids = [
            random.randint(1, vocab_size - 1)
            for _ in range(prompt_len)
        ]

        req = InferenceRequest(
            request_id=request_id,
            prompt_ids=prompt_ids,
            max_new_tokens=5,
        )

        requests.append(req)

    # ------------------------------------------------------------
    # 5. 分发请求
    # ------------------------------------------------------------

    dispatcher = RoundRobinDispatcher(num_workers=num_workers)

    for req in requests:
        worker_id = dispatcher.dispatch(req)

        print(
            f"[Main] dispatch request {req.request_id} "
            f"to worker {worker_id}",
            flush=True,
        )

        request_queues[worker_id].put(req)

    # ------------------------------------------------------------
    # 6. 收集结果
    # ------------------------------------------------------------

    results: dict[int, InferenceResponse] = {}

    while len(results) < len(requests):
        try:
            resp = response_queue.get(timeout=30)
        except queue.Empty:
            raise RuntimeError("Timeout waiting for worker response")

        assert isinstance(resp, InferenceResponse)

        results[resp.request_id] = resp

        print(
            f"[Main] got response for request {resp.request_id} "
            f"from worker {resp.worker_id}, "
            f"generated_len={len(resp.generated_ids)}",
            flush=True,
        )

    # ------------------------------------------------------------
    # 7. 通知 worker 退出
    # ------------------------------------------------------------

    for q in request_queues:
        q.put(None)

    for p in workers:
        p.join()

    # ------------------------------------------------------------
    # 8. 打印最终结果
    # ------------------------------------------------------------

    print("\n========== Final Results ==========")

    for request_id in sorted(results.keys()):
        resp = results[request_id]

        print(
            f"request_id={request_id}, "
            f"worker_id={resp.worker_id}, "
            f"generated_ids={resp.generated_ids}"
        )


if __name__ == "__main__":
    # 多进程启动方式。
    #
    # spawn 更安全，尤其是在 CUDA 场景下。
    #
    # Linux 默认可能是 fork，但 CUDA + fork 容易产生问题，
    # 所以这里显式设置 spawn。
    mp.set_start_method("spawn", force=True)

    main()