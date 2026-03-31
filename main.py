"""Demo entrypoint: capture an FX graph, lower to Graph IR, and print nodes.

This script attempts to import PyTorch and use `torch.fx.symbolic_trace` to
produce a graph. If PyTorch is not available, it prints an informative message.
"""

from __future__ import annotations

from typing import Any


from lowering.graph_to_tensor_ir import lower_fx_to_graph_ir
from frontend.fx_capture import capture_with_shape
from models.mlp import SimpleMLP
from passes.dead_code_elimination import run_dead_code_elimination
from passes.op_fusion import run_op_fusion
from lowering.graph_to_tensor_ir import lower_graph_ir_to_tensor_ir


def _capture_fx_graph() -> Any:
    try:
        import torch
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required to capture an FX graph. Please install torch.") from e

    model = SimpleMLP(in_features=4, hidden=4, out_features=4)
    example_input = torch.randn(1, 4)
    gm = capture_with_shape(model, example_input)
    return gm, model, example_input


def main() -> int:
    try:
        gm, model, example_input = _capture_fx_graph()
    except RuntimeError as e:
        print(str(e))
        return 1

    ir = lower_fx_to_graph_ir(gm)

    print("\n=== Graph IR (before passes) ===")
    for n in ir:
        print(f"{n.name:16} | {n.op_type:12} | inputs={n.inputs} | shape={n.shape}")

    # Compact metadata summary before fusion
    print("\n=== Graph IR metadata (compact) ===")
    for n in ir:
        m = getattr(n, "meta", {}) or {}
        if any(k in m for k in ("fx_op", "target", "module_type", "source_kind")):
            print(f"  {n.name:16} fx_op={m.get('fx_op'):12} target={m.get('target'):10} module_type={m.get('module_type')}")

    # Run passes: dead code elimination, then op fusion
    ir_dce = run_dead_code_elimination(ir)
    ir_fused = run_op_fusion(ir_dce)

    print("\n=== Graph IR (after passes) ===")
    for n in ir_fused:
        print(f"{n.name:16} | {n.op_type:12} | inputs={n.inputs} | shape={n.shape}")

    # Lower to Tensor IR
    from ir.tensor_ir import TensorIR

    tir = lower_graph_ir_to_tensor_ir(ir_fused)
    print("\n=== Tensor IR operations ===")
    for op in tir:
        print(f"{op.name:20} | {op.op_type:12} | inputs={op.inputs} | output={op.output} | shape={op.shape}")

    # Print attribute summaries for linear-like ops
    print("\n=== Tensor IR attribute summaries ===")
    for op in tir:
        if op.op_type in ("operation", "linear_relu"):
            wshape = op.attributes.get("weight_shape") or getattr(op.attributes.get("weight"), "shape", None)
            bshape = op.attributes.get("bias_shape") or getattr(op.attributes.get("bias"), "shape", None)
            dtype = op.attributes.get("dtype")
            if wshape or bshape or dtype:
                print(f"  {op.name:20} weight_shape={wshape:16} bias_shape={bshape:12} dtype={dtype}")

    # Run simple memory planner
    from memory.planner import plan_memory

    plan = plan_memory(tir)
    print("\n=== Memory planning ===")
    print("Lifetimes:")
    for name, (s, e) in plan["lifetimes"].items():
        print(f"  {name:20} start={s:2}, end={e:2}")
    print(f"Estimated peak memory: {plan['peak_bytes']} bytes")
    if plan["reused"]:
        print("Reused buffers:")
        for tensor, buf in plan["reused"]:
            print(f"  {tensor} -> buffer {buf} (size={plan['buffers'].get(buf)})")
    else:
        print("No buffer reuse opportunities found")

    # Generate toy accelerator instructions
    from backend.accel_backend import emit_instructions

    instrs = emit_instructions(tir)
    print("\n=== Toy accelerator instructions ===")
    for i in instrs:
        print(i)

    # Run NumPy backend where possible
    try:
        import numpy as np
        from backend.numpy_backend import run_tensor_ir

        # Build input mapping: map TensorIR inputs to example arrays
        inputs = {}
        # Heuristic: find first op inputs that are not produced by any op
        produced = {op.output for op in tir}
        for op in tir:
            for inp in op.inputs:
                if inp not in produced and inp not in inputs:
                    # try to get actual example input value from example_input (torch tensor)
                    if hasattr(example_input, "numpy") and inp == "x":
                        inputs[inp] = example_input.numpy().astype(np.float32)
                    else:
                        # create a random example matching op.shape if available
                        # fallback to (1,4)
                        shape = op.shape or (1, 4)
                        inputs[inp] = np.random.randn(*shape).astype(np.float32)

        output = run_tensor_ir(tir, inputs)
        print("\nNumPy backend output shape:", getattr(output, "shape", None))
        print(f"NumPy backend output summary: min={float(output.min()):.6f}, max={float(output.max()):.6f}")

        # Compare NumPy backend output to PyTorch model output on the same example input
        try:
            import torch

            model.eval()
            with torch.no_grad():
                pt_out = model(example_input)
            # Convert pt_out to numpy
            pt_arr = pt_out.cpu().numpy()
            # Compare shapes and max abs diff
            same_shape = pt_arr.shape == output.shape
            max_abs_diff = float(np.max(np.abs(pt_arr - output)))
            print("\nPyTorch output shape:", pt_arr.shape)
            print(f"Max abs difference between PyTorch and NumPy backend: {max_abs_diff:.6e}")
            print("Shapes match:", same_shape, "Close:", max_abs_diff < 1e-4)
        except Exception as e:
            print("PyTorch comparison failed:", e)
    except Exception as e:
        print("NumPy backend not available or failed:", e)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
