"""Simple operator fusion pass.

Detects a pattern where a linear-like op is immediately followed by a ReLU
and fuses them into a single `linear_relu` op when safe:
- the ReLU consumes a single producer
- the linear's output is only used by that ReLU

This is conservative and intentionally simple.
"""

from __future__ import annotations

from typing import Dict, List

from lowering.graph_to_tensor_ir import GraphIR, GraphIRNode


def run_op_fusion(ir: GraphIR) -> GraphIR:
    # Build a usage map: producer name -> list of consumer node names
    usage: Dict[str, List[str]] = {}
    name_to_node: Dict[str, GraphIRNode] = {n.name: n for n in ir}
    for n in ir:
        for inp in n.inputs:
            usage.setdefault(inp, []).append(n.name)

    # Identify fusion candidates (producer -> relu consumer) conservatively
    fusion_candidates: Dict[str, str] = {}  # producer_name -> fused_name
    fused_nodes_meta: Dict[str, Dict] = {}

    for n in ir:
        # Identify ReLU-like consumer nodes using metadata rather than name
        # Consumer must be an operation and have single input
        if n.op_type != "operation" or len(n.inputs) != 1:
            continue

        cons_meta = getattr(n, "meta", {}) or {}
        is_relu = False
        # Prefer explicit module_type if available
        mtype = cons_meta.get("module_type")
        if isinstance(mtype, str) and mtype.lower() == "relu":
            is_relu = True
        # Fallback: check fx_op/target metadata for functional relu
        fx_op = cons_meta.get("fx_op")
        target = cons_meta.get("target")
        if not is_relu and fx_op in ("call_function", "call_method") and target is not None:
            try:
                if "relu" in str(target).lower():
                    is_relu = True
            except Exception:
                pass

        if not is_relu:
            continue

        prod_name = n.inputs[0]
        prod = name_to_node.get(prod_name)
        if prod is None:
            continue
        # Determine whether producer is linear-like by metadata
        prod_meta = getattr(prod, "meta", {}) or {}
        prod_mtype = prod_meta.get("module_type")
        is_linear_like = False
        if isinstance(prod_mtype, str) and prod_mtype.lower() in ("linear", "linearmodule"):
            is_linear_like = True
        # Also allow producer that has weight in meta
        if not is_linear_like and prod_meta.get("weight") is not None:
            is_linear_like = True

        # Ensure producer is an operation and only used by this relu
        users = usage.get(prod_name, [])
        if prod.op_type == "operation" and len(users) == 1 and is_linear_like:
            # Safe to fuse: record mapping
            fused_name = f"{prod.name}_{n.name}_fused"
            fusion_candidates[prod_name] = fused_name
            # store metadata for creating fused node later
            fused_nodes_meta[prod_name] = {
                "fused_name": fused_name,
                "producer": prod,
                "consumer": n,
            }

    if not fusion_candidates:
        return ir

    # Build new node list: replace producer+consumer with fused node (placed at producer position)
    new_nodes: List[GraphIRNode] = []
    # Map to replace old names (consumer names) with fused names
    replace_map: Dict[str, str] = {}

    for n in ir:
        if n.name in fusion_candidates:
            # This node is a producer to be fused; create fused node here
            meta = fused_nodes_meta[n.name]
            prod = meta["producer"]
            cons = meta["consumer"]
            fused_name = meta["fused_name"]
            fused_inputs = list(prod.inputs)
            fused_shape = cons.shape or prod.shape
            fused_meta = {**prod.meta, **cons.meta}
            fused_node = GraphIRNode(name=fused_name, op_type="linear_relu", inputs=fused_inputs, shape=fused_shape, meta=fused_meta)
            new_nodes.append(fused_node)
            # remember to replace consumer name with fused_name in downstream nodes
            replace_map[cons.name] = fused_name
            # Also ensure producer and consumer are not added separately
            continue

        # Skip consumer nodes that were fused
        if n.name in replace_map and n.name != replace_map.get(n.name, n.name):
            # if somehow mapping exists, skip
            continue

        # For other nodes, copy but apply replacement to inputs
        new_n = GraphIRNode(name=n.name, op_type=n.op_type, inputs=[replace_map.get(x, x) for x in n.inputs], shape=n.shape, meta=n.meta)
        new_nodes.append(new_n)

    # It's possible that some replacements refer to names not yet updated in inputs; do a final pass
    for node in new_nodes:
        node.inputs = [replace_map.get(x, x) for x in node.inputs]

    # Build new IR
    new_ir = GraphIR()
    for node in new_nodes:
        new_ir.add_node(node)

    return new_ir
