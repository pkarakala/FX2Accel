"""Simple Dead Code Elimination (DCE) for GraphIR.

This pass removes nodes that do not contribute to any `output` nodes.
It preserves `input` and `output` nodes.
"""

from __future__ import annotations

from typing import Set

from lowering.graph_to_tensor_ir import GraphIR, GraphIRNode


def run_dead_code_elimination(ir: GraphIR) -> GraphIR:
    # Find names of nodes that are outputs
    output_nodes = [n for n in ir if n.op_type == "output"]
    reachable: Set[str] = set()

    # Start from outputs and walk backwards following inputs
    stack = []
    for out in output_nodes:
        reachable.add(out.name)
        for inp in out.inputs:
            stack.append(inp)

    while stack:
        name = stack.pop()
        if name in reachable:
            continue
        reachable.add(name)
        node = ir.get(name)
        if node is None:
            continue
        for inp in node.inputs:
            if inp not in reachable:
                stack.append(inp)

    # Always preserve inputs and outputs
    for n in ir:
        if n.op_type == "input":
            reachable.add(n.name)

    # Construct new IR with nodes in original order but filtered
    new_ir = GraphIR()
    for n in ir:
        if n.name in reachable:
            new_ir.add_node(n)

    return new_ir
