# SGLang Tutor

这个仓库用于学习和讲解 [SGLang](https://github.com/sgl-project/sglang) 的实现原理。仓库中引用了 SGLang 上游项目的原版代码，但用途不是维护 SGLang 本身，而是围绕源码阅读、运行链路和核心模块设计进行教学整理。

## 仓库内容

- `python/`：从 SGLang 原仓库保留的 Python 版本相关源码，供教学文档引用和源码阅读使用。
- `sgl-kernel/`：SGLang Python runtime 会直接引用的 kernel 代码与 Python 包，例如 `sgl_kernel` 和 `torch.ops.sgl_kernel.*`。
- `rust/sglang-grpc/`：`python/pyproject.toml` 中声明的 Rust gRPC 扩展源码，用于保留 Python 包的源码构建路径。
- `proto/sglang/runtime/v1/sglang.proto`：`rust/sglang-grpc` 构建时使用的 protobuf 定义。
- `learning/`：本仓库维护的教学材料。
- `LICENSE`：保留 SGLang 上游项目的开源许可证信息。

为了让仓库更聚焦，当前版本没有保存 SGLang 的完整工程内容，例如上游 Docker、benchmark、测试集、文档站点、模型网关和第三方依赖目录等。`python/sglang` 内部的测试与 benchmark 辅助包、`sgl-kernel` 内部的 benchmark/tests 也已移除，仅保留 Python runtime 和源码构建需要的 kernel 源码。需要完整代码、安装方式或最新开发信息时，请访问 SGLang 官方仓库。

## 教学文件

主要教学文档位于：

```text
learning/sglang-source-reading/
```

当前包含：

```text
learning/sglang-source-reading/README.md
learning/sglang-source-reading/00-feature-map.md
learning/sglang-source-reading/01-request-lifecycle.md
learning/sglang-source-reading/02-scheduler-core.md
learning/sglang-source-reading/03-kv-cache-radix-cache.md
```

建议阅读顺序：

1. `learning/sglang-source-reading/README.md`
2. `learning/sglang-source-reading/00-feature-map.md`
3. `learning/sglang-source-reading/01-request-lifecycle.md`
4. `learning/sglang-source-reading/02-scheduler-core.md`
5. `learning/sglang-source-reading/03-kv-cache-radix-cache.md`

## 与 SGLang 原项目的关系

SGLang 的版权和许可证归原项目贡献者所有。本仓库只在教学目的下保留必要的 Python 源码片段和阅读材料，并尽量使用相对路径引用源码，避免依赖某台机器上的本地路径或不稳定的代码行号。

如果你想运行、部署或参与开发 SGLang，请以官方仓库为准：

```text
https://github.com/sgl-project/sglang
```
