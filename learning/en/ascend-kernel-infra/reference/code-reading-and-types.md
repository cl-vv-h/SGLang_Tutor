# Code Reading Manual: Types, Shapes & Addresses

## Type Distinctions

| Type System | Example | What It Is |
|---|---|---|
| Python object | `torch.Tensor` | Host-side tensor handle (metadata + data pointer) |
| Triton IR type | `tl.tensor` | Triton's internal representation (≠ torch.Tensor) |
| Pointer | `tl.tensor.to(pointer_type)` | Raw memory address in Triton |
| Value block | `tl.tensor` after load | Data loaded into registers/UB |

## Key Distinction: tl.tensor ≠ torch.Tensor

```python
# torch.Tensor: host-side, contains metadata + GPU data pointer
x = torch.randn(256, device="cuda")

# tl.tensor: device-side, inside Triton kernel
@triton.jit
def kernel(x_ptr, ...):
    # x_ptr is a pointer to GM
    data = tl.load(x_ptr + offsets)  # data is a tl.tensor (register/UB value)
```

## Ascend C Type Hierarchy

| Type | Memory Location | Usage |
|---|---|---|
| `GlobalTensor<T>` | GM (HBM) | Typed view of global memory |
| `LocalTensor<T>` | UB (on-chip) | Typed view of unified buffer |
| `T*` (raw pointer) | GM | Raw address (used with offset arithmetic) |

## Why Pointer + Offset Works

```cpp
GlobalTensor<float> A;  // Base address in GM
int offset = 128;
// A[offset] is valid because GlobalTensor overloads operator[]
// Equivalent to: *(reinterpret_cast<__gm__ float*>(A.GetPhyAddr()) + offset)
```

## C++ Namespaces: `at::`, `c10::`, `torch::`, and Ascend-Related Prefixes

Reading an Ascend C operator means reading more than its Device kernel. A complete operator also contains PyTorch registration, a Host wrapper, torch_npu stream adaptation, a CANN launch path, and the Device implementation. That is why several namespaces appear along one `.cpp` call chain.

### 1. What `::` Means

A C++ **namespace** is a scope that organizes names and prevents symbol collisions:

```cpp
at::Tensor logits;
std::int64_t rows = logits.size(0);
AscendC::GlobalTensor<half> logitsGm;
```

Read these expressions one layer at a time:

- `at::Tensor` looks up the type `Tensor` in namespace `at`;
- `std::int64_t` looks up `int64_t` in the C++ standard-library namespace;
- `AscendC::GlobalTensor` looks up a class template in namespace `AscendC`;
- `AscendC::GlobalTensor<half>` instantiates that template with element type `half`.

`::` is the **scope-resolution operator**. It does not access data inside an object. The `.` in `logits.size(0)` invokes a member on an object, while `ptr->method()` invokes a member through a pointer. A namespace does not allocate memory, construct an object, or by itself tell you whether code runs on the CPU or NPU.

The real sgl-kernel-npu source uses nested project namespaces:

```cpp
namespace sglang {
namespace npu_kernel {
at::Tensor apply_token_bitmask(at::Tensor logits);
}  // namespace npu_kernel
}  // namespace sglang
```

C++17 also permits the equivalent compact form:

```cpp
namespace sglang::npu_kernel {
at::Tensor apply_token_bitmask(at::Tensor logits);
}
```

The fully qualified function name is `sglang::npu_kernel::apply_token_bitmask`. This isolates the project symbol from functions with the same short name; it does not imply that `sglang` is a separate process or runtime.

### 2. Namespace-to-Layer Map

| Spelling | Definition / Ownership | Main Layer | Typical Symbols | Reading Rule |
|---|---|---|---|---|
| `std::` | C++ standard library | Host C++ and the supported Device C++ subset | `std::min`, `std::tuple`, `std::apply` | General language library; not owned by PyTorch, CANN, or sgl-kernel-npu |
| `at::` | PyTorch ATen tensor library | PyTorch / Host | `at::Tensor`, `at::ScalarType`, `at::empty`, `at::zeros` | High-level tensor handles and operators; storage may be on the NPU even though the C++ handle is used on the Host |
| `c10::` | PyTorch C10 core library | PyTorch core / Host | `c10::optional`, `c10::Device`, `c10::DeviceType`, `c10::Scalar`, `c10::Stream` | Backend-neutral core types, device/stream abstractions, and utilities |
| `torch::` | PyTorch C++ frontend and registration API | PyTorch / Host | `torch::Library`, `torch::Tensor`, `torch::nn` | Entry points intended for C++ users and extension authors |
| `c10_npu::` | torch_npu NPU core adaptation | torch_npu / Host | `c10_npu::getCurrentNPUStream()` and NPU guards | Specializes PyTorch device and stream abstractions for the NPU backend; it is a top-level namespace, not `c10::npu` |
| `at_npu::native::` | torch_npu native operator/execution implementation | torch_npu / Host | `at_npu::native::OpCommand` | Organizes NPU Host execution logic; it is not an Ascend C Device namespace |
| `platform_ascendc::` | CANN Ascend C Host platform API | CANN / Host tiling | `PlatformAscendCManager`, `CoreMemType::UB` | Queries core counts and memory capacities for Host tiling; it does not execute on an AI Core |
| `AscendC::` | CANN Ascend C programming API | Primarily Device kernel code | `GlobalTensor`, `LocalTensor`, `TPipe`, `TQue`, `DataCopy`, `Add` | Device abstractions for GM/local buffers, movement, computation, and synchronization |
| `sglang::npu_kernel::` | sgl-kernel-npu project code | Depends on the definition; often Host wrapper code | `apply_token_bitmask`, `TorchNpuHelper` | Shows source ownership and C++ scope, not execution placement by itself |

The central boundary is:

```text
at:: / c10:: / torch::        generic PyTorch Host layer
             |
c10_npu:: / at_npu::native:: torch_npu NPU Host adaptation
             |
platform_ascendc:: / ACLRT    CANN Host query and launch layer
             |
AscendC::                     Ascend C Device kernel API
```

These are not four competing Tensor APIs. They are different abstractions required at different boundaries of one invocation.

### 3. What `at::Tensor` Actually Represents

ATen, short for “A Tensor Library,” is PyTorch's foundational tensor and operator library. `at::Tensor` is its core C++ tensor type. The object stores or references:

- metadata such as shape, stride, dtype, device, and layout;
- a reference-counted handle to the underlying `TensorImpl` and storage;
- framework state related to autograd and version tracking.

If `logits.device()` is an NPU, the elements live in NPU Global Memory. The Host-side `at::Tensor logits` is a lightweight handle that manages metadata and storage lifetime. Copying the handle normally increases a shared reference count; it does not copy every NPU element.

The fixed source contains this statement:

```cpp
workingLogits = at::zeros({numIndices, paddedVocabSize}, selectedLogits.options());
```

| Expression | Type / Role |
|---|---|
| `workingLogits` | An `at::Tensor` Host handle declared earlier |
| `at::zeros(...)` | An ATen factory operator that creates and zero-initializes a tensor |
| `{numIndices, paddedVocabSize}` | The new tensor shape |
| `selectedLogits.options()` | Carries dtype, device, layout, and other creation options |
| Resulting storage | Located on the NPU when the inherited options select the NPU |

This statement may dispatch NPU allocation and zeroing work, but the result is not an `AscendC::LocalTensor` and does not place data in one AI Core's UB.

### 4. Are `torch::Tensor` and `at::Tensor` Different Containers?

In normal LibTorch and PyTorch C++ extension headers, namespace `torch` exposes ATen declarations. Consequently, `torch::Tensor` and `at::Tensor` normally resolve to the same underlying Tensor type rather than two containers requiring a data conversion. PyTorch's [`torch/types.h`](https://github.com/pytorch/pytorch/blob/main/torch/csrc/api/include/torch/types.h#L11-L55) does this with `namespace torch { using namespace at; }`.

That does not make every `torch::` API identical to every `at::` API:

- `at::empty` and `at::zeros` emphasize the lower-level ATen tensor/operator interface;
- `torch::nn` and `torch::optim` are higher-level C++ frontend facilities;
- `torch::Library` is a custom-operator registration handle, not a tensor;
- `torch::Library` and `at::Tensor` appear together because they serve different responsibilities.

Python `torch.Tensor` crosses a binding/dispatcher boundary before C++ receives it as an `at::Tensor` or `torch::Tensor` handle. Do not interpret the two C++ spellings as a Python-versus-C++ object conversion.

### 5. Read `c10::optional<at::Tensor>` from the Outside In

The real Host signature is:

```cpp
HOST_API at::Tensor apply_token_bitmask(
    at::Tensor logits,
    at::Tensor bitmask,
    c10::optional<at::Tensor> indices);
```

Decompose `c10::optional<at::Tensor>` as follows:

1. `at::Tensor` is the contained value type when a value exists;
2. `c10::optional<T>` means that a value of type `T` may or may not be present;
3. the complete type therefore means “an optional indices tensor.”

The fixed source checks and normalizes it with real C++ syntax:

```cpp
bool hasIndices =
    indices.has_value() && indices->defined() && indices->numel() > 0;
at::Tensor rowIndices;
if (hasIndices) {
    rowIndices = indices.value().to(at::kLong).contiguous();
}
```

In current public PyTorch headers, `c10::optional` exposes `std::optional` for compatibility. Older versions and existing projects still use the C10 spelling, so do not mechanically rename fixed-version code.

- `has_value()` only says whether the optional contains a value;
- `value()` or `*indices` obtains the contained Tensor handle;
- no optional value is not the same as a present tensor with shape `[0]`;
- a present value still needs `defined()`, `numel()`, dtype, device, and shape checks.

Different namespaces compose normally in C++ templates: C10 supplies the optional-value abstraction, while ATen supplies the Tensor type.

### 6. `c10_npu::` and `at_npu::native::`

Underscores and scope operators matter:

```cpp
c10::Device                         // Device inside namespace c10
c10_npu::getCurrentNPUStream()      // function inside top-level c10_npu
at::Tensor                          // Tensor inside namespace at
at_npu::native::OpCommand           // OpCommand inside at_npu::native
```

`c10_npu` and `at_npu` are torch_npu namespaces. Their names indicate proximity to the C10 device layer and the ATen-native operator layer, but the compiler treats `c10` and `c10_npu` as unrelated top-level namespace names.

The fixed [`torch_helper.h#L122-L133`](https://github.com/sgl-project/sgl-kernel-npu/blob/d5630dff41c8108216f835597e63f6d3a7445908/csrc/utils/torch_helper.h#L122-L133) contains:

```cpp
auto acl_stream = c10_npu::getCurrentNPUStream().stream(false);
at_npu::native::OpCommand::RunOpApi(#kernel_name, acl_call);
```

These statements are inside the `EXEC_KERNEL_CMD` macro. `#kernel_name` is preprocessor **stringification**, which turns a macro argument into a string; it is not a normal unary operator that can be used in an arbitrary function body.

The first line obtains the current PyTorch NPU stream and its lower-level CANN handle. The second integrates the launch callable with torch_npu execution and error handling. Both run in the Host process; neither creates another process.

### 7. `AscendC::` Versus `platform_ascendc::`

The Host source queries platform properties:

```cpp
auto ascendcPlatform =
    platform_ascendc::PlatformAscendCManager::GetInstance();
uint64_t ubSize = 0;
ascendcPlatform->GetCoreMemSize(
    platform_ascendc::CoreMemType::UB, ubSize);
```

`platform_ascendc::` provides Host platform and tiling facilities. The CPU queries UB capacity and uses the returned scalar to calculate `tileLength`. `ubSize` is not UB storage, and the query does not allocate a Device buffer.

The Device source uses:

```cpp
AscendC::LocalTensor<T> logitsLocal =
    inQueueLogits.AllocTensor<T>();
AscendC::DataCopy(
    logitsLocal, logitsGm[logitsGmOffset], curTileLen);
```

Here `AscendC::` names the Device API. `LocalTensor` is a local-buffer view; `DataCopy` is the operation that describes actual movement. Compare the real [`op_host` lines 94–113](https://github.com/sgl-project/sgl-kernel-npu/blob/d5630dff41c8108216f835597e63f6d3a7445908/csrc/apply_token_bitmask/op_host/apply_token_bitmask.cpp#L94-L113) with the [`op_kernel` lines 82–91](https://github.com/sgl-project/sgl-kernel-npu/blob/d5630dff41c8108216f835597e63f6d3a7445908/csrc/apply_token_bitmask/op_kernel/apply_token_bitmask.cpp#L82-L91).

The substring “ascendc” is therefore not sufficient evidence for execution placement. Check the target, directory, function attributes, ABI parameters, and actual APIs.

### 8. Why `aclrtStream` Does Not Use `aclrt::`

Many CANN runtime interfaces are C-style APIs. C has no C++ namespaces, so libraries use name prefixes:

- `acl` is the general AscendCL/ACL prefix;
- `aclrt...` identifies runtime types and functions such as `aclrtStream`;
- `aclrtlaunch_<kernel>` is a generated kernel launch stub;
- `ACLRT_LAUNCH_KERNEL(...)` is a launch macro.

`aclrtStream` is one complete C-style type name, not `Stream` inside namespace `aclrt`. Likewise, `TORCH_CHECK`, `EXEC_KERNEL_CMD`, and `ACLRT_LAUNCH_KERNEL` are preprocessor macros and do not use namespace lookup.

### 9. The `npu` in `TORCH_LIBRARY_FRAGMENT(npu, m)`

The following code uses three separate naming mechanisms:

```cpp
namespace sglang::npu_kernel { /* C++ namespace */ }

TORCH_LIBRARY_FRAGMENT(npu, m) {
    m.def("apply_token_bitmask(Tensor logits, Tensor bitmask, "
          "Tensor? indices=None) -> Tensor");
}

// Python
torch.ops.npu.apply_token_bitmask(logits, bitmask, indices)
```

- `sglang::npu_kernel` is a compile-time C++ namespace;
- the macro argument `npu` is a logical PyTorch dispatcher/operator-library namespace;
- `torch.ops.npu` is Python's dynamic access path to that dispatcher namespace;
- no C++ `namespace npu {}` is required;
- multiple extension libraries may register into the same dispatcher namespace, so it is not proof of repository ownership.

The registration source also wraps code in `namespace { ... }`. This is an **anonymous namespace**, which gives helper symbols translation-unit-local visibility. It is unrelated to the dispatcher namespace named `npu`.

### 10. One Real Call Chain

Read the real `apply_token_bitmask` path in source order:

| Source Expression | Layer | Responsibility |
|---|---|---|
| `c10::optional<at::Tensor> indices` | PyTorch Host type contract | Receives a possibly absent Tensor argument |
| `at::zeros(..., selectedLogits.options())` | ATen / Host | Creates a working tensor using the input options |
| `platform_ascendc::...GetCoreMemSize(...)` | CANN Host platform | Queries UB capacity for Host tile calculation |
| `c10_npu::getCurrentNPUStream()` | torch_npu Host core | Gets the current PyTorch NPU stream |
| `sglang::npu_kernel::TorchNpuHelper::ConvertTypes(...)` | Project Host wrapper | Converts Tensor handles to launch device addresses |
| `ACLRT_LAUNCH_KERNEL(...)` | CANN Runtime Host API | Submits kernel work to the stream |
| `at_npu::native::OpCommand::RunOpApi(...)` | torch_npu Host framework | Integrates launch and backend error handling |
| `AscendC::GlobalTensor<T>` | Ascend C Device | Binds a received GM address as a typed view |
| `AscendC::DataCopy(...)` | Ascend C Device | Moves data between GM and a local buffer |

A useful source-reading rule is: circle every `A::B`, find which header and repository define `A`, and then verify whether the statement belongs to a Host or Device target. A namespace is a clue, not final proof.

### 11. Namespaces in More Complex Operators

Attention, MatMul, DeepEP, and mega-kernel code introduces additional namespaces:

| Spelling | Ownership and Responsibility | Typical Side |
|---|---|---|
| `ge::` | CANN Graph Engine types such as `ge::DataType`, `ge::Format`, and `ge::graphStatus` | Host operator description, validation, and status |
| `gert::` | CANN Graph Engine Runtime descriptors and contexts such as `gert::Tensor`, `gert::StorageShape`, and tiling contexts | Host runtime / tiling |
| `optiling::` | Common organizational namespace for operator tiling classes, validation, and tiling-data helpers | Primarily Host; shared/generated tiling structures may also be included by Device headers |
| `matmul_tiling::` | CANN MatMul tiling helpers that derive matrix execution parameters from shape, format, dtype, and hardware | Host tiling |
| `host_utils::`, `ge_helper::` | sgl-kernel-npu project helpers for dtype conversion, hashing, and GE/GERT descriptors | Project Host code; not universal CANN namespaces |
| `pybind11::` | Third-party C++/Python binding library | Host Python-extension boundary |
| `deep_ep::` | sgl-kernel-npu DeepEP project namespace, including `Buffer`, `Config`, and `EventHandle` | Ownership marker; individual functions may be Host wrappers or launches |
| `pto::` | Low-level Device tile/instruction helper used by the current mega kernel | Device; its `GlobalTensor` is not the same class as `AscendC::GlobalTensor` |
| `AscendC::MicroAPI::` | Lower-level Ascend C Vector register/micro-instruction API, including `RegTensor` and `MaskReg` | Device Vector path |

Names that contain “Tensor” are not interchangeable. `gert::Tensor`, `at::Tensor`, `AscendC::GlobalTensor<T>`, and `pto::GlobalTensor<...>` are respectively a runtime descriptor, a PyTorch Host handle, an Ascend C GM view, and a PTO Device view.

For an unknown `foo::Bar`:

1. inspect the current file's `#include` directives;
2. use “go to definition” or search for `namespace foo`;
3. determine whether the file is compiled into a Host or Device target;
4. classify `Bar` as a handle, descriptor, tiling helper, Device view, enum, or other object.

### 12. `using namespace`, `using`, and Namespace Aliases

Source does not always show the complete qualified name:

```cpp
using std::string;
using namespace matmul_tiling;
namespace py = pybind11;
```

- `using std::string;` introduces only `string` into the current scope;
- `using namespace matmul_tiling;` makes names from that namespace participate in unqualified lookup;
- `namespace py = pybind11;` creates an alias, so `py::arg` and `pybind11::arg` name the same declaration.

Lexical nesting can also hide a prefix:

```cpp
namespace AscendC {
MicroAPI::MaskReg mask = MicroAPI::CreateMask<int32_t>();
}
```

Because this declaration is already inside `namespace AscendC`, `MicroAPI::` resolves to the complete namespace `AscendC::MicroAPI::`. This is why one file may use both short and fully qualified forms.

These declarations do not copy implementations or perform runtime conversions. When reading code, search upward for `using` declarations and enclosing namespaces. When writing public headers, prefer fully qualified names or a precise `using A::B` to reduce ambiguity.

### 13. Definition and Source Entry Points

- [PyTorch ATen C++ API](https://docs.pytorch.org/cppdocs/api/aten/index.html)
- [PyTorch C10 C++ API](https://docs.pytorch.org/cppdocs/api/c10/index.html)
- [PyTorch Torch Library API](https://docs.pytorch.org/cppdocs/api/library/index.html)
- [Fixed `apply_token_bitmask` Host wrapper](https://github.com/sgl-project/sgl-kernel-npu/blob/d5630dff41c8108216f835597e63f6d3a7445908/csrc/apply_token_bitmask/op_host/apply_token_bitmask.cpp)
- [Fixed `torch_helper.h`](https://github.com/sgl-project/sgl-kernel-npu/blob/d5630dff41c8108216f835597e63f6d3a7445908/csrc/utils/torch_helper.h)
- [Fixed `apply_token_bitmask` Device kernel](https://github.com/sgl-project/sgl-kernel-npu/blob/d5630dff41c8108216f835597e63f6d3a7445908/csrc/apply_token_bitmask/op_kernel/apply_token_bitmask.cpp)

Use versions of PyTorch, torch_npu, CANN, and sgl-kernel-npu that match the deployment environment. Namespace responsibilities are comparatively stable, but header paths, aliases, and internal APIs can change.

### 14. Checkpoints and Detailed Answers

#### 1. Why may `c10::optional<at::Tensor>` contain two namespaces?

**Answer:** A C++ template can accept any type that satisfies its requirements. `c10::optional<T>` expresses that a value of type `T` may exist; `at::Tensor` is the Tensor handle type substituted for `T`. Namespaces uniquely identify the two types but do not require the outer and inner types to come from one library.

#### 2. Does `at::Tensor x = y;` copy all NPU elements?

**Answer:** Normally no. It copies a Tensor handle, and `x` and `y` share the reference-counted implementation/storage. Whether new element storage is created depends on an actual operation such as `clone()`, some `contiguous()` calls, `.to()`, or an explicit `copy_`.

#### 3. How can a Host-side `at::Tensor` represent NPU data?

**Answer:** Handle location and element-storage location are separate concerns. The Host object carries device, shape, stride, and lifetime metadata while its storage may reside in NPU GM. The Host wrapper validates the contract and extracts a device address; the Device kernel accesses elements through that address.

#### 4. Does `platform_ascendc::CoreMemType::UB` create UB on the Host?

**Answer:** No. It is an enum value used to query target-hardware UB capacity. The result helps Host tiling. Actual on-chip buffers are arranged by Device mechanisms such as `TPipe`, `TQue`, and `TBuf` together with the compiler.

#### 5. Are `torch.ops.npu.foo`, `c10_npu::foo`, and `namespace npu {}` equivalent?

**Answer:** No. The first is Python access to a dispatcher namespace, the second is a qualified C++ name in the top-level namespace `c10_npu`, and the third declares an actual C++ namespace named `npu`. They use different naming systems.

#### 6. Why can the substring `AscendC` not determine Host versus Device placement?

**Answer:** `platform_ascendc::` is a Host platform-query interface, while `AscendC::` primarily contains Device programming APIs. Host code may also include Ascend C tiling types. Use the source directory, build target, function attributes, ABI, and called APIs together.
