# 13. SGLang NPU 多场景模型运行手册

这一讲是场景化启动手册。前面的章节分别讲环境、参数、PD、LoRA、profiling；这一讲把它们组合成“我要跑某类模型/某种部署形态时，该用什么命令、先验收什么、常见坑在哪”。

本讲默认你已经完成：

- 宿主机 `npu-smi info` 正常。
- 官方 Docker 或自建环境可以执行 `sglang serve --help`。
- 模型已经放在 `/workspace/sglang-npu/models` 或 `/home/{myspace}/sglang-npu-workspace/models`。
- 普通单卡请求已经能返回。

## 场景总览

```mermaid
flowchart TD
  A["SGLang NPU 跑模型"] --> B["模型来源"]
  A --> C["卡数/并行"]
  A --> D["服务形态"]
  A --> E["模型能力"]
  A --> F["性能/稳定性模式"]

  B --> B1["在线模型名"]
  B --> B2["ModelScope 本地模型"]
  B --> B3["Hugging Face 本地模型"]
  B --> B4["离线拷贝模型"]

  C --> C1["单卡"]
  C --> C2["多卡 TP"]
  C --> C3["多实例多端口"]

  D --> D1["普通混合 serving"]
  D --> D2["PD 分离"]
  D --> D3["后台 Docker 服务"]

  E --> E1["Dense LLM"]
  E --> E2["MoE"]
  E --> E3["LoRA"]
  E --> E4["量化"]
  E --> E5["多模态"]

  F --> F1["长上下文"]
  F --> F2["Graph 开关"]
  F --> F3["Profiling"]
```

## 0. 统一约定

### 0.1 Docker 内路径

官方 Docker 推荐把个人工作目录挂载到容器：

```bash
-v /home/{myspace}/sglang-npu-workspace:/workspace/sglang-npu
```

本讲默认容器内路径：

```bash
export WORKSPACE=/workspace/sglang-npu
export MODEL_ROOT=/workspace/sglang-npu/models
export LOG_ROOT=/workspace/sglang-npu/logs
mkdir -p "$MODEL_ROOT" "$LOG_ROOT"
```

### 0.2 通用 NPU 参数

绝大多数场景都建议显式带上：

```bash
--device npu \
--attention-backend ascend
```

单卡时通常加：

```bash
--base-gpu-id 0 \
--tp-size 1
```

多卡时通常加：

```bash
--tp-size <N>
```

> SGLang 里不少参数名沿用 CUDA 术语，例如 `--disable-cuda-graph`、`--cuda-graph-max-bs`。在 NPU 下它们会映射到 NPU graph 语义，不代表真的使用 CUDA。

### 0.3 最小验收命令

每个场景启动后都先做：

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "用一句话说明你已经在 NPU 上运行。"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

日志里至少确认：

```text
device=npu
attention_backend=ascend
```

## 1. 按模型来源区分

### 1.1 在线模型名启动

适合：临时验证、网络可访问模型源、模型不大。

```bash
sglang serve \
  --model-path Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --base-gpu-id 0 \
  --tp-size 1
```

优点是命令短。缺点是下载来源、缓存目录、网络速度和 token 权限会影响启动时间，不适合严肃 benchmark。

### 1.2 ModelScope 本地模型启动

适合：国内网络、内网模型仓库、可控缓存路径。

先下载：

```bash
python3 -m pip install -U modelscope
export MODELSCOPE_CACHE=/workspace/sglang-npu/cache/modelscope
mkdir -p "$MODELSCOPE_CACHE" "$MODEL_ROOT"

modelscope download \
  --model Qwen/Qwen2.5-7B-Instruct \
  --local_dir "$MODEL_ROOT/Qwen2.5-7B-Instruct"
```

启动：

```bash
sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --base-gpu-id 0 \
  --tp-size 1
```

如果 ModelScope CLI 版本不支持 `--local_dir`，用 Python API：

```bash
python3 - <<'PY'
from modelscope import snapshot_download

snapshot_download(
    "Qwen/Qwen2.5-7B-Instruct",
    local_dir="/workspace/sglang-npu/models/Qwen2.5-7B-Instruct",
    cache_dir="/workspace/sglang-npu/cache/modelscope",
)
PY
```

### 1.3 Hugging Face 本地模型启动

适合：能访问 Hugging Face，或团队已有 HF 镜像源。

```bash
export HF_HOME=/workspace/sglang-npu/cache/huggingface
export HF_TOKEN=<your_token_if_needed>

huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir "$MODEL_ROOT/Qwen2.5-7B-Instruct"

sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --base-gpu-id 0 \
  --tp-size 1
```

### 1.4 完全离线模型启动

适合：生产环境、封闭内网、无法访问外部模型源。

在可联网机器下载模型后，把完整目录拷贝到：

```text
/home/{myspace}/sglang-npu-workspace/models/Qwen2.5-7B-Instruct
```

在容器内检查：

```bash
ls "$MODEL_ROOT/Qwen2.5-7B-Instruct"
find "$MODEL_ROOT/Qwen2.5-7B-Instruct" -maxdepth 1 -type f | sort | head -30
```

启动：

```bash
sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --base-gpu-id 0 \
  --tp-size 1
```

## 2. 按卡数和并行方式区分

### 2.1 单卡 Dense 模型

适合：最小功能验证、开发调试、单卡 profiling。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --base-gpu-id 0 \
  --tp-size 1 \
  2>&1 | tee "$LOG_ROOT/single-card.log"
```

验收重点：

- `torch.npu.device_count()` 至少为 1。
- 日志确认 `device=npu`。
- 单请求和 stream 请求都能返回。

### 2.2 多卡 Tensor Parallel

适合：模型单卡放不下，或希望提升吞吐。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-32B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --tp-size 4 \
  --base-gpu-id 0 \
  2>&1 | tee "$LOG_ROOT/tp4.log"
```

验收重点：

- 日志出现 HCCL 初始化。
- 每张卡显存都有占用。
- `tp_size` 和可见 NPU 数量一致。
- 单卡能跑通后再跑多卡。

常见问题：

| 现象 | 方向 |
|---|---|
| 多卡启动卡住 | HCCL、rank/device 映射、端口冲突。 |
| 单卡正常，多卡慢 | HCCL 通信占比高、TP size 不合适。 |
| 某张卡显存异常 | rank 绑定、模型 shard、可见设备配置。 |

### 2.3 多实例多端口

适合：一台机器上用不同 NPU 跑多个小模型，或做 A/B 实验。

实例 A：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --base-gpu-id 0 \
  --tp-size 1
```

实例 B：

```bash
ASCEND_RT_VISIBLE_DEVICES=1 \
sglang serve \
  --model-path "$MODEL_ROOT/another-model" \
  --host 0.0.0.0 \
  --port 8001 \
  --device npu \
  --attention-backend ascend \
  --base-gpu-id 0 \
  --tp-size 1
```

注意：`ASCEND_RT_VISIBLE_DEVICES=1` 后，进程内看到的第一张卡通常仍是逻辑 0，所以 `--base-gpu-id 0` 是常见写法。

## 3. 按服务形态区分

### 3.1 普通混合 Prefill/Decode Serving

这是默认形态，prefill 和 decode 在同一个 server 中完成。

```bash
sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --tp-size 1
```

适合：

- 单机服务。
- 初次验证。
- 大多数开发调试。

### 3.2 PD 分离：Prefill Server

适合：长 prompt 多、decode 持续时间长、希望 prefill/decode 分开扩容。

Prefill 侧：

```bash
export ASCEND_MF_STORE_URL="tcp://127.0.0.1:18000"
# Atlas 800I A2 且要走 RDMA 时再按团队规范启用：
# export ASCEND_MF_TRANSFER_PROTOCOL="device_rdma"

sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8100 \
  --device npu \
  --attention-backend ascend \
  --tp-size 1 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend ascend \
  --disaggregation-bootstrap-port 8995
```

### 3.3 PD 分离：Decode Server

Decode 侧：

```bash
export ASCEND_MF_STORE_URL="tcp://127.0.0.1:18000"
# export ASCEND_MF_TRANSFER_PROTOCOL="device_rdma"

sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8200 \
  --device npu \
  --attention-backend ascend \
  --tp-size 1 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend ascend \
  --pd-disaggregation
```

PD 验收重点：

- `memfabric_hybrid` 已安装。
- Prefill 和 Decode 都能看到同一模型。
- `ASCEND_MF_STORE_URL` 一致。
- 日志里出现 Ascend transfer engine 初始化。
- 先本机双进程验证，再扩展到多机。

### 3.4 后台 Docker 服务

适合：长时间运行，或者把服务交给其他人调用。

```bash
docker run -d \
  --name sglang-npu-qwen-{myspace} \
  --restart unless-stopped \
  --privileged \
  --network=host \
  --ipc=host \
  --shm-size=16g \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/hisi_hdc \
  -v /usr/local/sbin:/usr/local/sbin:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /var/queue_schedule:/var/queue_schedule \
  -v /home/{myspace}/sglang-npu-workspace:/workspace/sglang-npu \
  docker.io/lmsysorg/sglang:main-cann8.5.0-910b \
  sglang serve \
    --model-path /workspace/sglang-npu/models/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --device npu \
    --attention-backend ascend \
    --base-gpu-id 0 \
    --tp-size 1
```

查看：

```bash
docker logs -f sglang-npu-qwen-{myspace}
```

停止：

```bash
docker stop sglang-npu-qwen-{myspace}
docker rm sglang-npu-qwen-{myspace}
```

## 4. 按模型能力区分

### 4.1 MoE 模型

适合：Qwen-MoE、DeepSeek-MoE 等带 expert 的模型。

```bash
sglang serve \
  --model-path "$MODEL_ROOT/moe-model" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --tp-size 4
```

MoE 验收重点：

- routing 正确，输出不异常。
- 多卡时 expert 相关通信稳定。
- profiling 中关注 routed expert、shared expert、top-k、combine。
- 如果出现 fallback，确认是正确性兜底还是性能降级。

### 4.2 LoRA 模型

适合：一个 base model 加多个 adapter。

先确认本地 SGLang 版本参数：

```bash
sglang serve --help | grep -i lora
```

常见启动模板：

```bash
sglang serve \
  --model-path "$MODEL_ROOT/base-model" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --tp-size 1 \
  --enable-lora \
  --max-loras-per-batch 4
```

LoRA 场景的参数在不同版本中可能还涉及 adapter 路径、LoRA 名称映射、rank 限制等，请以 `sglang serve --help` 和当前分支文档为准。

验收重点：

- base-only 输出正常。
- 单 adapter 输出正常。
- 多 adapter batch 输出正常。
- 日志确认 Ascend LoRA backend 或 NPU LoRA kernel 被使用。

### 4.3 量化模型

适合：AWQ、GPTQ、W8A8、W4A4 等量化模型。

先确认模型目录里有量化配置：

```bash
ls "$MODEL_ROOT/quant-model"
grep -R "quant" "$MODEL_ROOT/quant-model"/*.json || true
```

启动模板：

```bash
sglang serve \
  --model-path "$MODEL_ROOT/quant-model" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --tp-size 1
```

如果你的 SGLang 版本需要显式量化参数，先查：

```bash
sglang serve --help | grep -i quant
```

量化验收重点：

- 输出正确性和基线模型一致性。
- 日志中是否走 NPU quant backend。
- 是否 fallback 到慢路径。
- HBM 是否明显下降。

### 4.4 多模态模型

适合：Qwen-VL、GLM-4.6V 等多模态模型。

```bash
sglang serve \
  --model-path "$MODEL_ROOT/Qwen3-VL-30B-A3B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --tp-size 4 \
  --enable-multimodal
```

多模态验收重点：

- 文本-only 请求先跑通。
- 图片请求再跑通。
- 关注 processor 是否走了 NPU 适配 patch。
- 注意图像分辨率和 token 数会显著影响 prefill。

## 5. 按性能/稳定性模式区分

### 5.1 长上下文模型

适合：长 prompt、RAG、文档问答。

```bash
sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --tp-size 1 \
  --chunked-prefill-size 4096
```

验收重点：

- 4096/8192 prompt 不 OOM。
- prefill latency 可接受。
- KV cache 显存占用符合预期。
- 如果首 token 很慢，先 profile prefill。

### 5.2 关闭 Graph 做定位

适合：遇到 graph capture 卡住、replay 错误、shape 不稳定。

```bash
sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --tp-size 1 \
  --disable-cuda-graph
```

如果关闭 graph 后正常，说明问题可能在 capture/replay、shape key、静态输入地址或 graph 覆盖范围。

### 5.3 调整 Graph 覆盖范围

适合：decode batch size 稳定，但 graph replay 覆盖不足。

```bash
sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --tp-size 1 \
  --cuda-graph-max-bs 64
```

验收重点：

- TPOT 是否下降。
- graph capture 时间是否可接受。
- HBM 是否增长过多。
- 是否频繁 fallback 到 eager。

### 5.4 Profiling 模式

适合：要生成 profiling 跑测报告。

先按第 12 讲准备：

```bash
export RUN_ID=$(date +%Y%m%d-%H%M%S)
export PROF_ROOT=/workspace/sglang-npu/profiling-runs/${RUN_ID}
mkdir -p "$PROF_ROOT"
```

启动：

```bash
sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --tp-size 1 \
  2>&1 | tee "$PROF_ROOT/server.log"
```

压测：

```bash
python3 /workspace/sglang-npu/bench_openai.py \
  --prompt-len 512 \
  --max-tokens 128 \
  --concurrency 8 \
  --requests 64 \
  --out "$PROF_ROOT/decode-c8"
```

## 6. 组合场景示例

### 6.1 本地 ModelScope + 单卡 + profiling

```bash
export MODEL_PATH=/workspace/sglang-npu/models/Qwen2.5-7B-Instruct
export RUN_ID=$(date +%Y%m%d-%H%M%S)
export PROF_ROOT=/workspace/sglang-npu/profiling-runs/${RUN_ID}
mkdir -p "$PROF_ROOT"

sglang serve \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --tp-size 1 \
  2>&1 | tee "$PROF_ROOT/server.log"
```

### 6.2 离线模型 + TP4 + graph

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3

sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-32B-Instruct" \
  --host 0.0.0.0 \
  --port 8000 \
  --device npu \
  --attention-backend ascend \
  --tp-size 4 \
  --cuda-graph-max-bs 64 \
  2>&1 | tee "$LOG_ROOT/tp4-graph.log"
```

### 6.3 PD 分离 + 本地模型 + 双进程

先启动 prefill：

```bash
export ASCEND_MF_STORE_URL="tcp://127.0.0.1:18000"

sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8100 \
  --device npu \
  --attention-backend ascend \
  --tp-size 1 \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend ascend \
  --disaggregation-bootstrap-port 8995
```

再启动 decode：

```bash
export ASCEND_MF_STORE_URL="tcp://127.0.0.1:18000"

sglang serve \
  --model-path "$MODEL_ROOT/Qwen2.5-7B-Instruct" \
  --host 0.0.0.0 \
  --port 8200 \
  --device npu \
  --attention-backend ascend \
  --tp-size 1 \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend ascend \
  --pd-disaggregation
```

## 7. 场景选择速查表

| 目标 | 建议场景 |
|---|---|
| 第一次验证 NPU | 单卡 Dense + 本地模型。 |
| 网络可用，快速试跑 | 在线模型名启动。 |
| 国内下载模型 | ModelScope 本地模型。 |
| 生产稳定 | 离线模型 + 后台 Docker。 |
| 模型太大 | 多卡 TP。 |
| 长 prompt 多 | 长上下文 + chunked prefill。 |
| Decode 压力大 | graph 覆盖 + profiling。 |
| prefill/decode 资源不均 | PD 分离。 |
| 多 adapter 服务 | LoRA。 |
| expert 模型 | MoE。 |
| 降低显存或带宽 | 量化模型。 |
| 图片/多模态 | 多模态模型。 |

## 8. 通用排错顺序

1. 宿主机 `npu-smi info` 是否正常。
2. 容器内 `npu-smi info` 是否正常。
3. `torch.npu.is_available()` 是否为 `True`。
4. `import sglang, sgl_kernel_npu` 是否成功。
5. 日志里是否是 `device=npu`。
6. 日志里 attention backend 是否是 `ascend`。
7. 单卡是否正常。
8. 再看多卡、PD、LoRA、MoE、量化、多模态。
9. 出现性能问题再进入第 12 讲 profiling。

## 本讲小结

SGLang NPU 跑模型可以按四个维度组合：

- 模型来源：在线、ModelScope、本地 HF、离线。
- 计算资源：单卡、多卡 TP、多实例。
- 服务形态：普通 serving、PD 分离、后台服务。
- 模型能力：Dense、MoE、LoRA、量化、多模态、长上下文。

初学时不要一上来组合所有能力。最稳的路线是：**单卡 Dense 本地模型 -> 多卡 TP -> 长上下文/graph -> PD/LoRA/MoE/量化/多模态 -> profiling 与性能优化**。
