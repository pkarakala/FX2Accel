"""Simple NumPy backend to execute TensorIR operations.

Assumptions/simplifications:
- Tensors are NumPy arrays.
- `TensorOp.attributes` may contain parameter arrays under keys like
  `weight` and `bias`. If present, `operation` or `linear_relu` will use
  these for computation. If absent, the backend performs conservative
  passthrough behavior so pipelines and tests can exercise lowering
  without requiring parameter extraction.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ir.tensor_ir import TensorIR, TensorOp


def run_tensor_ir(tir: TensorIR, inputs: Dict[str, np.ndarray], params: Dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Execute the TensorIR and return the final output tensor.

    Args:
        tir: the TensorIR to execute
        inputs: mapping from input tensor names to numpy arrays
        params: optional mapping for parameter tensors (by name)
    Returns:
        numpy array representing the final output tensor (last op's output)
    """
    env: Dict[str, np.ndarray] = {}
    if params:
        env.update(params)

    # Seed inputs
    env.update(inputs)

    last_out = None
    for op in tir:
        # Gather operand arrays
        operands = [env.get(n) for n in op.inputs]

        if op.op_type == "linear_relu":
            # Use canonical weight layout: (out_features, in_features)
            w = op.attributes.get("weight")
            b = op.attributes.get("bias")
            if w is not None:
                x = operands[0]
                # x: (batch, in), w: (out, in) -> x.dot(w.T) -> (batch, out)
                y = x.dot(w.T)
                if b is not None:
                    y = y + b
                y = np.maximum(y, 0)
            else:
                x = operands[0]
                y = np.maximum(x, 0)
        elif op.op_type == "relu":
            x = operands[0]
            y = np.maximum(x, 0)
        else:
            # generic 'operation': attempt linear if weight attr present
            w = op.attributes.get("weight")
            b = op.attributes.get("bias")
            if w is not None:
                x = operands[0]
                # canonical layout: w (out, in)
                y = x.dot(w.T)
                if b is not None:
                    y = y + b
            else:
                # Conservative default: if a single input, pass it through;
                # if multiple inputs, try elementwise add when arrays available.
                if len(operands) == 1:
                    y = operands[0]
                else:
                    # elementwise add where possible
                    y = None
                    for arr in operands:
                        if arr is None:
                            continue
                        if y is None:
                            y = arr.copy()
                        else:
                            y = y + arr
                    if y is None:
                        # final fallback: create zeros of expected shape if available
                        if op.shape:
                            y = np.zeros(op.shape, dtype=np.float32)
                        else:
                            raise RuntimeError(f"Cannot compute op {op.name}: missing inputs")

        env[op.output] = y
        last_out = y

    if last_out is None:
        raise RuntimeError("No output produced by TensorIR")
    return last_out
