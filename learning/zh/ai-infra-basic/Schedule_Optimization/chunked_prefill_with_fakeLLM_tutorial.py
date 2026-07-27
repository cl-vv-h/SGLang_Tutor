# chunked_prefill_tutorial.py

"""
============================================================
Chunked Prefill 教学版源码
============================================================

本代码目标：
    用一个简化的 LLM 推理调度器，讲清楚 Chunked Prefill 的核心原理。

重点不是实现真实 Transformer，而是实现推理调度逻辑：

    1. 请求 Request 的状态管理；
    2. Prefill 阶段如何分 chunk；
    3. Decode 阶段如何优先调度；
    4. token budget 如何限制每轮 batch 的大小；
    5. KV Cache 如何按 position 写入；
    6. Prefill 完成后如何进入 Decode；
    7. Decode 每轮如何生成 1 个 token。

============================================================
背景：Prefill / Decode
============================================================

一个 LLM 请求通常包含两个阶段：

1. Prefill:
    输入 prompt token。
    一次性计算 prompt 中所有 token 的 KV Cache。

2. Decode:
    每次生成一个新 token。
    每次只输入上一步生成的新 token，
    并使用历史 KV Cache。

============================================================
普通 Prefill 的问题
============================================================

如果一个请求 prompt 很长：

    prompt_len = 10000

普通 prefill 会一次性处理 10000 个 token。

这会导致：

    1. 长 prefill 长时间占用 GPU；
    2. 已经在 decode 的请求被阻塞；
    3. 用户流式输出卡顿；
    4. Inter-Token Latency 变差。

============================================================
Chunked Prefill
============================================================

Chunked Prefill 将长 prompt 切成多个小 chunk：

    prompt_len = 10000
    chunk_size = 2048

切成：

    chunk0: token 0 ~ 2047
    chunk1: token 2048 ~ 4095
    chunk2: token 4096 ~ 6143
    ...

每一轮调度只处理一小段 prefill，
这样 decode 请求可以插入到 prefill chunk 之间。

============================================================
核心调度策略
============================================================

每一轮调度：

    1. 先调度 decode 请求；
    2. 每个 decode 请求消耗 1 个 token budget；
    3. 如果还有剩余 token budget，再调度 prefill chunk；
    4. prefill chunk 的长度受以下因素限制：
        - max_prefill_chunk_size
        - request 剩余 prompt 长度
        - 当前剩余 token budget

============================================================
注意
============================================================

这份代码是教学版，不是真实 LLM engine。

真实系统中还会涉及：

    1. GPU kernel；
    2. PagedAttention；
    3. Block KV Cache；
    4. Continuous batching；
    5. CUDA Graph；
    6. Tensor Parallel；
    7. Prefix Cache；
    8. Speculative Decoding；
    9. Prefill/Decode 分离部署。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import random


# ============================================================
# 1. 请求状态定义
# ============================================================

class RequestStatus(str, Enum):
    """
    请求状态。

    WAITING_PREFILL:
        请求刚进来，还没有完成 prompt prefill。

    PREFILLING:
        请求正在进行 prefill。
        对于 chunked prefill 来说，一个请求可能经历多次 PREFILLING。

    DECODING:
        prompt 已经 prefill 完成，正在逐 token decode。

    FINISHED:
        请求已经生成完成。
    """

    WAITING_PREFILL = "WAITING_PREFILL"
    PREFILLING = "PREFILLING"
    DECODING = "DECODING"
    FINISHED = "FINISHED"


# ============================================================
# 2. Request 数据结构
# ============================================================

@dataclass
class Request:
    """
    一个推理请求。

    request_id:
        请求 ID。

    prompt_ids:
        prompt token ids。
        shape 可以理解为：
            [prompt_len]

    max_new_tokens:
        最多生成多少个新 token。

    status:
        当前请求状态。

    num_prefilled:
        已经完成 prefill 的 prompt token 数量。

        对于普通 prefill：
            num_prefilled 会从 0 一次性变成 prompt_len。

        对于 chunked prefill：
            num_prefilled 会逐步增加：
                0 -> 4 -> 8 -> 12 -> ...

    generated_ids:
        decode 阶段生成的新 token。

    kv_cache_positions:
        模拟该请求已经写入 KV Cache 的 position。

        真实系统里 KV Cache 是张量：
            K.shape = [num_layers, num_heads, seq_len, head_dim]
            V.shape = [num_layers, num_heads, seq_len, head_dim]

        教学版只记录 position，用来表示哪些 token 已经写入 KV Cache。
    """

    request_id: int
    prompt_ids: List[int]
    max_new_tokens: int

    status: RequestStatus = RequestStatus.WAITING_PREFILL
    num_prefilled: int = 0
    generated_ids: List[int] = field(default_factory=list)
    kv_cache_positions: List[int] = field(default_factory=list)

    @property
    def prompt_len(self) -> int:
        """
        prompt 总长度。
        """
        return len(self.prompt_ids)

    @property
    def remaining_prefill_tokens(self) -> int:
        """
        还没有 prefill 的 prompt token 数量。
        """
        return self.prompt_len - self.num_prefilled

    @property
    def is_prefill_done(self) -> bool:
        """
        prompt 是否已经全部 prefill。
        """
        return self.num_prefilled >= self.prompt_len

    @property
    def num_generated(self) -> int:
        """
        已经生成的新 token 数。
        """
        return len(self.generated_ids)

    @property
    def is_decode_done(self) -> bool:
        """
        decode 是否完成。
        """
        return self.num_generated >= self.max_new_tokens

    @property
    def current_context_len(self) -> int:
        """
        当前上下文长度。

        包括：
            prompt token
            已生成 token

        注意：
            在 prefill 尚未完成时，当前实际可用上下文是 num_prefilled。
            在 decode 阶段，prompt 已经全部 prefill。
        """

        if self.status in (
            RequestStatus.WAITING_PREFILL,
            RequestStatus.PREFILLING,
        ):
            return self.num_prefilled

        return self.prompt_len + self.num_generated


# ============================================================
# 3. 调度任务类型
# ============================================================

class ScheduledTaskType(str, Enum):
    """
    当前调度 step 中的任务类型。

    PREFILL:
        处理一段 prompt chunk。

    DECODE:
        生成一个新 token。
    """

    PREFILL = "PREFILL"
    DECODE = "DECODE"


@dataclass
class ScheduledTask:
    """
    一个被调度到当前 batch 的任务。

    task_type:
        PREFILL 或 DECODE。

    request:
        该任务属于哪个请求。

    token_ids:
        本次要送入模型的 token ids。

        如果是 prefill：
            token_ids 是一个 prompt chunk。
            shape 可以理解为 [chunk_len]

        如果是 decode：
            token_ids 通常只有 1 个 token。
            shape 可以理解为 [1]

    start_pos:
        本次 token 在整个 request sequence 中的起始 position。

        prefill chunk:
            start_pos = request.num_prefilled

        decode:
            start_pos = prompt_len + num_generated - 1
            表示当前 decode token 对应的位置。

    例如：

        prompt_len = 10
        已经生成 2 个 token

        当前 decode 的新输入 token 是上一步生成的 token，
        它要写入 KV Cache 的 position = 10 + 2 - 1 = 11
    """

    task_type: ScheduledTaskType
    request: Request
    token_ids: List[int]
    start_pos: int

    @property
    def num_tokens(self) -> int:
        """
        当前 task 消耗多少 token budget。
        """
        return len(self.token_ids)


# ============================================================
# 4. KV Cache Manager 教学模拟
# ============================================================

class KVCacheManager:
    """
    教学版 KV Cache Manager。

    真实系统中的 KV Cache 是大张量，例如：

        K_cache[layer][request][head][position][head_dim]
        V_cache[layer][request][head][position][head_dim]

    在本教学代码中，为了聚焦调度逻辑，
    只记录每个 request 已经写入了哪些 position。

    例如：

        request 0 prompt_len=8

        完成 chunk0 后：
            positions = [0,1,2,3]

        完成 chunk1 后：
            positions = [0,1,2,3,4,5,6,7]

        decode 一个 token 后：
            positions = [0,1,2,3,4,5,6,7,8]
    """

    def __init__(self):
        self.cache: Dict[int, List[int]] = {}

    def allocate_for_request(self, request: Request):
        """
        为请求初始化 KV Cache 记录。
        """
        self.cache[request.request_id] = []

    def write_tokens(
        self,
        request: Request,
        start_pos: int,
        num_tokens: int,
    ):
        """
        模拟写入一段 token 的 K/V。

        request:
            当前请求。

        start_pos:
            写入起始 position。

        num_tokens:
            写入 token 数量。

        写入 positions:
            start_pos, start_pos + 1, ..., start_pos + num_tokens - 1
        """

        positions = list(range(start_pos, start_pos + num_tokens))

        self.cache[request.request_id].extend(positions)

        # 同步到 request 上，方便打印观察。
        request.kv_cache_positions.extend(positions)

    def get_positions(self, request: Request) -> List[int]:
        """
        返回某个请求已经写入的 KV Cache positions。
        """
        return self.cache.get(request.request_id, [])


# ============================================================
# 5. Fake LLM Model
# ============================================================

class FakeLLMModel:
    """
    一个假的 LLM 模型。

    它不做真实 Transformer 计算，
    只模拟两个行为：

        1. prefill:
            输入一段 token，产生这些 token 的 KV Cache。
            教学版中由 KVCacheManager 写 position 表示。

        2. decode:
            输入一个 token，生成下一个 token。
            教学版中随机生成一个 token id。

    真实 LLM 中：

        Prefill 输入：
            input_ids.shape = [batch, prefill_len]

        Prefill 输出：
            logits.shape = [batch, prefill_len, vocab_size]
            KV Cache 写入所有 prefill positions

        Decode 输入：
            input_ids.shape = [batch, 1]

        Decode 输出：
            logits.shape = [batch, 1, vocab_size]
            KV Cache 追加一个 position
    """

    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def forward_prefill(
        self,
        request: Request,
        token_ids: List[int],
        start_pos: int,
        kv_manager: KVCacheManager,
    ):
        """
        模拟 prefill forward。

        token_ids:
            当前 prefill chunk。
            shape 可理解为 [chunk_len]

        start_pos:
            该 chunk 写入 KV Cache 的起始 position。

        行为：
            将 chunk 中每个 token 的 K/V 写入 KV Cache。
        """

        chunk_len = len(token_ids)

        kv_manager.write_tokens(
            request=request,
            start_pos=start_pos,
            num_tokens=chunk_len,
        )

    def forward_decode(
        self,
        request: Request,
        token_ids: List[int],
        start_pos: int,
        kv_manager: KVCacheManager,
    ) -> int:
        """
        模拟 decode forward。

        token_ids:
            decode 输入 token。
            通常只有 1 个。
            shape 可理解为 [1]

        start_pos:
            当前 decode token 写入 KV Cache 的 position。

        行为：
            1. 写入当前 token 的 K/V；
            2. 随机生成 next_token_id。

        真实模型中：
            会计算 logits，然后 sample / argmax 得到 next_token。
        """

        assert len(token_ids) == 1

        kv_manager.write_tokens(
            request=request,
            start_pos=start_pos,
            num_tokens=1,
        )

        next_token_id = random.randint(0, self.vocab_size - 1)

        return next_token_id


# ============================================================
# 6. Chunked Prefill Scheduler
# ============================================================

class ChunkedPrefillScheduler:
    """
    Chunked Prefill 调度器。

    核心参数：

        max_num_batched_tokens:
            每一轮调度最多处理多少 token。

        max_prefill_chunk_size:
            一个 prefill chunk 最多包含多少 token。

        decode_first:
            是否优先调度 decode 请求。

    核心策略：

        每一轮 schedule：

            1. token_budget = max_num_batched_tokens

            2. 如果 decode_first:
                   先把所有 DECODING 请求加入 batch
                   每个 decode 请求消耗 1 token

            3. 剩余 token budget 用于 prefill chunk

            4. 每个 prefill chunk 长度：
                   min(
                       request.remaining_prefill_tokens,
                       max_prefill_chunk_size,
                       remaining_token_budget
                   )

            5. 返回当前 batch 的 tasks
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        max_prefill_chunk_size: int,
        decode_first: bool = True,
    ):
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_prefill_chunk_size = max_prefill_chunk_size
        self.decode_first = decode_first

    def schedule(
        self,
        active_requests: List[Request],
    ) -> List[ScheduledTask]:
        """
        根据当前 active requests 生成一个 batch。

        active_requests:
            当前还没有完成的请求列表。

        返回：
            tasks:
                当前 step 要执行的任务列表。
        """

        token_budget = self.max_num_batched_tokens
        tasks: List[ScheduledTask] = []

        # ------------------------------------------------------------
        # Step 1: 优先调度 decode 请求
        # ------------------------------------------------------------
        #
        # Decode 阶段每个请求本轮只处理 1 个 token。
        #
        # 为什么优先 decode？
        #
        # 因为 decode 决定用户看到新 token 的速度。
        # 如果 decode 被长 prefill 阻塞，流式输出会卡顿。
        if self.decode_first:
            for req in active_requests:
                if token_budget <= 0:
                    break

                if req.status != RequestStatus.DECODING:
                    continue

                if req.is_decode_done:
                    continue

                # decode 输入 token 是什么？
                #
                # 如果已经生成过 token，则输入上一个生成 token。
                #
                # 如果刚刚 prefill 完成，还没有生成任何 token，
                # 那么第一次 decode 可以使用 prompt 的最后一个 token
                # 作为输入。
                if req.generated_ids:
                    decode_input = req.generated_ids[-1]
                    start_pos = req.prompt_len + req.num_generated - 1
                else:
                    decode_input = req.prompt_ids[-1]
                    start_pos = req.prompt_len - 1

                task = ScheduledTask(
                    task_type=ScheduledTaskType.DECODE,
                    request=req,
                    token_ids=[decode_input],
                    start_pos=start_pos,
                )

                tasks.append(task)
                token_budget -= 1

        # ------------------------------------------------------------
        # Step 2: 调度 prefill chunk
        # ------------------------------------------------------------
        #
        # 剩余 token budget 用于 prefill。
        #
        # 一个长 prompt 不会一次性加入 batch，
        # 而是切成 chunk。
        for req in active_requests:
            if token_budget <= 0:
                break

            if req.status not in (
                RequestStatus.WAITING_PREFILL,
                RequestStatus.PREFILLING,
            ):
                continue

            if req.is_prefill_done:
                continue

            remaining = req.remaining_prefill_tokens

            # 本轮 prefill chunk 长度。
            chunk_len = min(
                remaining,
                self.max_prefill_chunk_size,
                token_budget,
            )

            if chunk_len <= 0:
                continue

            start = req.num_prefilled
            end = start + chunk_len

            token_chunk = req.prompt_ids[start:end]

            task = ScheduledTask(
                task_type=ScheduledTaskType.PREFILL,
                request=req,
                token_ids=token_chunk,
                start_pos=start,
            )

            tasks.append(task)

            token_budget -= chunk_len

            # 标记该请求正在 prefill。
            req.status = RequestStatus.PREFILLING

        return tasks


# ============================================================
# 7. LLM Engine
# ============================================================

class SimpleLLMEngine:
    """
    一个极简 LLM 推理引擎。

    它包含：

        1. request 队列；
        2. scheduler；
        3. fake model；
        4. KV cache manager；
        5. step() 执行一轮调度和计算。

    每一轮 step：

        1. scheduler 选择 tasks；
        2. 对每个 prefill task 执行 fake prefill；
        3. 对每个 decode task 执行 fake decode；
        4. 更新 request 状态。
    """

    def __init__(
        self,
        scheduler: ChunkedPrefillScheduler,
        model: FakeLLMModel,
        kv_manager: KVCacheManager,
    ):
        self.scheduler = scheduler
        self.model = model
        self.kv_manager = kv_manager

        self.requests: List[Request] = []
        self.time_step = 0

    def add_request(self, request: Request):
        """
        添加一个新请求。
        """
        self.requests.append(request)
        self.kv_manager.allocate_for_request(request)

    def active_requests(self) -> List[Request]:
        """
        返回所有未完成请求。
        """
        return [
            req for req in self.requests
            if req.status != RequestStatus.FINISHED
        ]

    def step(self):
        """
        执行一轮调度和模型计算。
        """

        active = self.active_requests()

        if not active:
            return False

        tasks = self.scheduler.schedule(active)

        if not tasks:
            return False

        self._print_scheduled_tasks(tasks)

        for task in tasks:
            if task.task_type == ScheduledTaskType.PREFILL:
                self._execute_prefill(task)
            elif task.task_type == ScheduledTaskType.DECODE:
                self._execute_decode(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")

        self.time_step += 1

        return True

    def _execute_prefill(self, task: ScheduledTask):
        """
        执行 prefill chunk。
        """

        req = task.request
        chunk_len = task.num_tokens

        self.model.forward_prefill(
            request=req,
            token_ids=task.token_ids,
            start_pos=task.start_pos,
            kv_manager=self.kv_manager,
        )

        # 更新已经 prefill 的 token 数。
        req.num_prefilled += chunk_len

        # 如果 prompt 已经全部 prefill，则进入 decode 阶段。
        if req.is_prefill_done:
            req.status = RequestStatus.DECODING
        else:
            req.status = RequestStatus.PREFILLING

    def _execute_decode(self, task: ScheduledTask):
        """
        执行 decode 一个 token。
        """

        req = task.request

        next_token = self.model.forward_decode(
            request=req,
            token_ids=task.token_ids,
            start_pos=task.start_pos,
            kv_manager=self.kv_manager,
        )

        req.generated_ids.append(next_token)

        if req.is_decode_done:
            req.status = RequestStatus.FINISHED
        else:
            req.status = RequestStatus.DECODING

    def _print_scheduled_tasks(self, tasks: List[ScheduledTask]):
        """
        打印当前 step 的调度结果。
        """

        total_tokens = sum(t.num_tokens for t in tasks)

        print(f"\n=== Step {self.time_step} ===")
        print(f"Scheduled total tokens: {total_tokens}")

        for task in tasks:
            req = task.request

            if task.task_type == ScheduledTaskType.PREFILL:
                print(
                    f"  PREFILL req={req.request_id}, "
                    f"chunk_len={task.num_tokens}, "
                    f"positions=[{task.start_pos}, {task.start_pos + task.num_tokens - 1}], "
                    f"num_prefilled_before={req.num_prefilled}/{req.prompt_len}"
                )

            else:
                print(
                    f"  DECODE  req={req.request_id}, "
                    f"input_token={task.token_ids[0]}, "
                    f"write_pos={task.start_pos}, "
                    f"generated_before={req.num_generated}/{req.max_new_tokens}"
                )

    def print_request_states(self):
        """
        打印所有请求状态。
        """

        print("\n--- Request States ---")

        for req in self.requests:
            print(
                f"req={req.request_id}, "
                f"status={req.status}, "
                f"prefilled={req.num_prefilled}/{req.prompt_len}, "
                f"generated={req.num_generated}/{req.max_new_tokens}, "
                f"kv_positions={req.kv_cache_positions}"
            )


# ============================================================
# 8. 构造请求
# ============================================================

def make_prompt(length: int, start: int) -> List[int]:
    """
    构造一个假的 prompt token ids。

    length:
        prompt 长度。

    start:
        起始 token id。
    """

    return list(range(start, start + length))


# ============================================================
# 9. Demo 1: Chunked Prefill
# ============================================================

def demo_chunked_prefill():
    """
    演示 Chunked Prefill。

    配置：

        max_num_batched_tokens = 8
        max_prefill_chunk_size = 4

    含义：

        每一轮最多处理 8 个 token。
        单个 prefill chunk 最大 4 个 token。

    请求：

        req0:
            prompt_len = 3
            max_new_tokens = 4

        req1:
            prompt_len = 12
            max_new_tokens = 3

        req2:
            prompt_len = 5
            max_new_tokens = 2

    观察点：

        req1 的 prompt 很长，不会一次性 prefill 12 个 token；
        而是被切成多个 chunk，每次最多 4 个 token。

        当 req0 进入 decode 后，decode 会优先调度。
    """

    scheduler = ChunkedPrefillScheduler(
        max_num_batched_tokens=8,
        max_prefill_chunk_size=4,
        decode_first=True,
    )

    model = FakeLLMModel(vocab_size=100)
    kv_manager = KVCacheManager()

    engine = SimpleLLMEngine(
        scheduler=scheduler,
        model=model,
        kv_manager=kv_manager,
    )

    req0 = Request(
        request_id=0,
        prompt_ids=make_prompt(length=3, start=10),
        max_new_tokens=4,
    )

    req1 = Request(
        request_id=1,
        prompt_ids=make_prompt(length=12, start=100),
        max_new_tokens=3,
    )

    req2 = Request(
        request_id=2,
        prompt_ids=make_prompt(length=5, start=200),
        max_new_tokens=2,
    )

    engine.add_request(req0)
    engine.add_request(req1)
    engine.add_request(req2)

    print("\n==============================")
    print("Demo: Chunked Prefill")
    print("==============================")

    engine.print_request_states()

    while engine.step():
        engine.print_request_states()

    print("\nAll requests finished.")


# ============================================================
# 10. Demo 2: 对比普通 Prefill
# ============================================================

def demo_non_chunked_prefill():
    """
    演示非 Chunked Prefill。

    通过把 max_prefill_chunk_size 设置得很大，
    模拟普通 prefill：一个请求的 prompt 尽可能一次性处理。

    你可以和 demo_chunked_prefill 对比观察：

        非 chunked:
            长 prompt 更容易一次性占满 token budget。

        chunked:
            长 prompt 被限制为小块，decode 更容易插入。
    """

    scheduler = ChunkedPrefillScheduler(
        max_num_batched_tokens=16,
        max_prefill_chunk_size=10_000,
        decode_first=True,
    )

    model = FakeLLMModel(vocab_size=100)
    kv_manager = KVCacheManager()

    engine = SimpleLLMEngine(
        scheduler=scheduler,
        model=model,
        kv_manager=kv_manager,
    )

    req0 = Request(
        request_id=0,
        prompt_ids=make_prompt(length=3, start=10),
        max_new_tokens=4,
    )

    req1 = Request(
        request_id=1,
        prompt_ids=make_prompt(length=12, start=100),
        max_new_tokens=3,
    )

    req2 = Request(
        request_id=2,
        prompt_ids=make_prompt(length=5, start=200),
        max_new_tokens=2,
    )

    engine.add_request(req0)
    engine.add_request(req1)
    engine.add_request(req2)

    print("\n==============================")
    print("Demo: Non-Chunked Prefill")
    print("==============================")

    engine.print_request_states()

    while engine.step():
        engine.print_request_states()

    print("\nAll requests finished.")


# ============================================================
# 11. 主入口
# ============================================================

if __name__ == "__main__":
    random.seed(0)

    demo_chunked_prefill()

    demo_non_chunked_prefill()