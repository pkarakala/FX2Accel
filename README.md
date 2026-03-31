
# FX2Accel — a mini ML accelerator compiler

FX2Accel is a compact, educational ML compiler pipeline that captures PyTorch models
via FX, lowers to simple intermediate representations, runs a small set of
optimizations, performs basic memory planning, and emits both a NumPy-executable
Tensor IR and a toy accelerator instruction stream.

Why this project matters
------------------------
Modern ML systems rely on compiler toolchains to map high-level models onto
specialized accelerators and runtime stacks. FX2Accel demonstrates the core
building blocks of such toolchains in a small, readable codebase:

- capturing model structure with PyTorch FX and shape propagation
- lowering into a Graph IR and a Tensor IR
- lightweight optimization passes (metadata-driven fusion, dead-code elimination)
- simple memory planning and buffer reuse
- backend codegen: NumPy execution for validation and a toy accelerator emitter

This repo is useful for learning about ML compiler design, experimenting with
fusion and memory strategies, and prototyping accelerator backends.

Compiler pipeline
-----------------
High-level flow implemented in this repo:

PyTorch model
	-> FX graph capture (symbolic_trace + ShapeProp)
	-> Graph IR (nodes with metadata and parameter extraction)
	-> optimization passes (dead code elimination, metadata-based op fusion)
	-> Tensor IR (ordered tensor ops with attributes/parameters)
	-> memory planning (liveness, peak memory, buffer reuse)
	-> backends:
			 - NumPy backend (executes Tensor IR with real parameters)
			 - Toy accelerator codegen (emits pseudo-instructions)

Repository structure
--------------------
- `frontend/` — FX capture and shape propagation helpers
- `lowering/` — lowering from FX -> Graph IR -> Tensor IR
- `ir/` — IR data structures (`GraphIR`, `TensorIR`)
- `passes/` — optimization passes (fusion, dead-code elimination, constant folding)
- `memory/` — simple memory planner and buffer reuse
- `backend/` — `numpy_backend` (execution) and `accel_backend` (toy codegen)
- `models/` — small example models (e.g., `SimpleMLP`)
- `tests/` — test suite validating pipeline correctness
- `main.py` — demo runner that exercises the full pipeline

Implemented features
--------------------
- Graph IR with node metadata and parameter extraction for `nn.Linear`
- Tensor IR with canonical parameter attributes (weights in (out, in) layout)
- Metadata-driven op fusion (linear -> relu -> linear_relu)
- Dead code elimination (backward reachability)
- Simple memory planner with lifetime analysis and buffer reuse
- NumPy backend to execute Tensor IR and validate numerics against PyTorch
- Toy accelerator instruction generator (readable pseudo-assembly)

Example output
--------------
Below is a short, realistic sample produced by `main.py`.

Graph IR before passes:

```
x | input | inputs=[] | shape=(1, 4)
lin | operation | inputs=['x'] | shape=(1, 4)
relu | operation | inputs=['lin'] | shape=(1, 4)
out | operation | inputs=['relu'] | shape=(1, 4)
output | output | inputs=['out'] | shape=(1, 4)
```

Graph IR after passes:

```
x | input | inputs=[] | shape=(1, 4)
lin_relu_fused | linear_relu | inputs=['x'] | shape=(1, 4)
out | operation | inputs=['lin_relu_fused'] | shape=(1, 4)
output | output | inputs=['out'] | shape=(1, 4)
```

Tensor IR (ops):

```
lin_relu_fused | linear_relu | inputs=['x'] | output=lin_relu_fused | shape=(1, 4)
out | operation | inputs=['lin_relu_fused'] | output=out | shape=(1, 4)
```

Memory planner summary:

```
Lifetimes:
	lin_relu_fused: start=0, end=1
	out: start=1, end=1
Estimated peak memory: 32 bytes
```

Toy accelerator instructions (excerpt):

```
LOAD x
LINEAR_RELU lin_relu_fused <- x
STORE lin_relu_fused
MATMUL out <- lin_relu_fused
STORE out
```

PyTorch vs NumPy backend comparison:

```
PyTorch output shape: (1, 4)
NumPy backend output shape: (1, 4)
Max abs difference between PyTorch and NumPy backend: 0.0
```

How to run
----------
1. Create a Python environment and install requirements:

```sh
python3 -m pip install -r requirements.txt
```

2. Run the demo pipeline (prints IRs, planner, instructions, and comparison):

```sh
python3 main.py
```

3. Run tests:

```sh
python3 -m pytest -q
```

Current limitations
-------------------
- Narrow operator support: only a few ops (Linear, ReLU) are parameter-aware.
- Toy accelerator: instruction stream is illustrative and not executable on real hardware.
- Simple memory planner: first-fit reuse and simple peak estimation.
- Conservative fusion heuristics: safe but limited set of patterns.

Future work
-----------
- Add Conv2d/BatchNorm parameter extraction and lowering
- Broaden fusion patterns and use operator/type metadata robustly
- Replace heuristic allocator with a cost-based memory scheduler
- Emit real accelerator code or export to an external runtime/SDK

Contributing & license
----------------------
This repository is intended as an educational prototype. Contributions are
welcome — please open issues or PRs for new lowering rules, passes, or
backend targets.

