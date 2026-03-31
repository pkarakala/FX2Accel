"""Toy accelerator codegen: emits pseudo-instructions from TensorIR.

This backend doesn't execute code; it generates a readable list of
instructions (strings) representing a toy accelerator instruction stream.
"""

from __future__ import annotations

from typing import List

from ir.tensor_ir import TensorIR, TensorOp


def emit_instructions(tir: TensorIR) -> List[str]:
    instrs: List[str] = []

    for op in tir:
        # Load inputs
        for inp in op.inputs:
            instrs.append(f"LOAD {inp}")

        if op.op_type == "linear_relu":
            # Emit fused instruction where possible
            instrs.append(f"LINEAR_RELU {op.output} <- {' '.join(op.inputs)}")
        elif op.op_type == "relu":
            instrs.append(f"RELU {op.output} <- {' '.join(op.inputs)}")
        else:
            # generic op: attempt MATMUL then ADD semantics for linear-like
            # if attributes suggest a matmul; otherwise emit generic
            instrs.append(f"MATMUL {op.output} <- {' '.join(op.inputs)}")

        instrs.append(f"STORE {op.output}")

    return instrs
