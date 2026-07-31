# Triton-Ascend 01: Program, Grid, Tile & First Kernel

## Vector Add: Triton vs PyTorch

```python
# PyTorch (eager):
c = a + b  # Single operation, CPU dispatches to GPU

# Triton (kernel):
@triton.jit
def add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    c = a + b
    tl.store(c_ptr + offsets, c, mask=mask)

# Launch with grid
grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
add_kernel[grid](a, b, c, n, BLOCK_SIZE=256)
```

## What `@triton.jit` Actually Means

JIT means **Just-In-Time compilation**. It does not mean “compile while Triton is installed” or “recompile every arithmetic instruction.” It means that, after the Python process has started, Triton compiles a Device-kernel variant when that variant is first needed and reuses a matching compiled result later.

The lifecycle is:

```text
Python imports the module
  -> @triton.jit wraps the function as JITFunction
  -> no NPU kernel has been launched yet

add_kernel[grid](...)
  -> bind Python arguments
  -> derive signature, constexpr values, specialization and options
  -> look up the in-process variant cache
  -> on a miss, compiler.compile(...) checks the disk cache
  -> on a real miss, build TTIR and lower/compile it for Ascend
  -> compute the launch grid
  -> launch the CompiledKernel on the current NPU Stream
```

The decorator implementation returns a [`JITFunction`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/runtime/jit.py#L880-L936). The bracket syntax is significant: [`KernelInterface.__getitem__`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/runtime/jit.py#L411-L419) remembers the grid and enters [`JITFunction.run`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/runtime/jit.py#L709-L748). A top-level JIT kernel is therefore launched as `kernel[grid](...)`, not called as an ordinary Host Python function.

Common reasons for a new compiled variant include:

- a different pointer element dtype, such as FP16 versus BF16;
- a different `tl.constexpr`, such as another `BLOCK_SIZE`;
- backend specialization attributes such as selected integer values or alignment;
- another target/backend, compiler option, or cache-invalidating environment setting;
- changed kernel source or compile-time dependency.

A runtime shape change does **not** automatically imply recompilation. If `n_elements` is an ordinary runtime scalar, values 1000 and 2000 may share one binary and use masks for the tail. If the wrapper converts shape into a different `BLOCK_SIZE: tl.constexpr`, that meta-parameter selects another variant.

The numeric grid is normally computed for launch after variant selection. Merely changing the program count does not inherently require another binary; a grid function that also changes constexpr meta-parameters does.

There are also two cache levels:

1. `JITFunction` keeps compiled variants in the current process.
2. [`compiler.compile`](https://github.com/triton-lang/triton-ascend/blob/be90ac7e52267822c0ea83d20b705c1e4eaf586f/python/triton/compiler/compiler.py#L228-L393) uses a disk cache for IR, metadata, binaries, and launcher artifacts.

This is why a first call may include compilation and runtime initialization, while a hot call mostly contains argument binding, launch, and Device execution. It is also why benchmarks must warm up the exact dtype/meta-parameter variant before measuring steady-state latency.

Finally, JIT compiles the supported Triton Python DSL, not arbitrary Python. Inside the kernel, overloaded operations build Triton IR values; ordinary dynamic containers, reflection, and arbitrary Python libraries do not become Device code.

## Key Concepts

| Concept | Meaning | In Code |
|---|---|---|
| Program | The kernel function | `@triton.jit` decorated function |
| Grid | How many program instances (blocks) | `grid` function |
| Block | One instance of the program on one core | `pid = tl.program_id(0)` |
| Tile | Chunk of data processed by one block | `BLOCK_SIZE` elements |

## Triton → Ascend NPU Pipeline

```text
Python Triton kernel (.py)
  → Triton-Ascend compiler
    → TTIR (Triton IR)
    → MLIR (Ascend dialect)
    → NPU binary (.o)
  → Ascend runtime loads and executes
```

## Ascend Grid Strategy

On Ascend NPU, `grid` maps differently than CUDA:
- CUDA: grid → thread blocks on SMs
- Ascend: grid → cores on AI Core array

The Triton-Ascend compiler handles the mapping automatically.
