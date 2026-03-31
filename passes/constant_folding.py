"""Placeholder for constant folding pass.

At the moment this repository does not construct constant-only subgraphs,
so this module is intentionally a no-op placeholder. When constant subgraphs
are represented in the GraphIR (for example nodes representing `aten::add`
with constant operands), a simple evaluator could replace those subgraphs
with constant tensors and update downstream inputs.

This file documents that intent and provides a `run_constant_folding(ir)`
function that currently returns the IR unchanged.
"""

from __future__ import annotations

from lowering.graph_to_tensor_ir import GraphIR


def run_constant_folding(ir: GraphIR) -> GraphIR:
    # No constant-only subgraph support yet; return IR unchanged.
    return ir
