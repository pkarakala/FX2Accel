"""Lowering utilities: FX Graph -> Graph IR.

This module provides a small, readable lowering from PyTorch FX graphs
to a minimal Graph IR suitable for later passes.

The goal is correctness and clarity rather than full operator coverage.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


class GraphIRNode:
    def __init__(self, name: str, op_type: str, inputs: Optional[List[str]] = None, shape: Optional[Any] = None, meta: Optional[Dict[str, Any]] = None):
        self.name = name
        self.op_type = op_type
        self.inputs = inputs or []
        self.shape = shape
        self.meta = meta or {}

    def __repr__(self) -> str:  # pragma: no cover - small helper
        return f"GraphIRNode(name={self.name!r}, op_type={self.op_type!r}, inputs={self.inputs!r}, shape={self.shape!r})"


class GraphIR:
    def __init__(self) -> None:
        self.nodes: List[GraphIRNode] = []
        self._by_name: Dict[str, GraphIRNode] = {}

    def add_node(self, node: GraphIRNode) -> None:
        self.nodes.append(node)
        self._by_name[node.name] = node

    def get(self, name: str) -> Optional[GraphIRNode]:
        return self._by_name.get(name)

    def __iter__(self) -> Iterable[GraphIRNode]:
        return iter(self.nodes)


def _extract_shape_from_meta(node_meta: Dict[str, Any]) -> Optional[Any]:
    # FX commonly stores `tensor_meta` with shape/dtype. Keep this simple.
    if not node_meta:
        return None
    tm = node_meta.get("tensor_meta")
    if tm is not None:
        # tensor_meta may be an object with `.shape` (torch.Size) or a dict-like
        try:
            # If it's a torch.fx.experimental.proxy_tensor.shape or similar
            s = getattr(tm, "shape", None)
            if s is not None:
                # normalize to tuple
                try:
                    return tuple(s)
                except Exception:
                    return s
            # dict-like fallback
            if isinstance(tm, dict):
                sh = tm.get("shape")
                if sh is not None:
                    try:
                        return tuple(sh)
                    except Exception:
                        return sh
        except Exception:
            return None
    # fallback: raw shape field
    return node_meta.get("shape")


def lower_fx_to_graph_ir(fx_graph) -> GraphIR:
    """Lower an FX Graph (or GraphModule) to a GraphIR.

    Args:
        fx_graph: either a `torch.fx.GraphModule` or a `torch.fx.Graph`.

    Returns:
        GraphIR: a simple in-memory representation of nodes with inputs and shape metadata.
    """
    # Lazy import to avoid hard dependency unless used.
    try:
        import torch.fx as fx  # type: ignore
    except Exception:  # pragma: no cover - environment dependent
        fx = None

    # accept GraphModule or Graph
    graph = fx_graph.graph if hasattr(fx_graph, "graph") else fx_graph
    graph_module = fx_graph if hasattr(fx_graph, "graph") else None

    ir = GraphIR()

    for node in graph.nodes:
        op = node.op  # 'placeholder', 'call_module', 'call_function', 'output', etc.
        name = node.name

        if op == "placeholder":
            op_type = "input"
            inputs: List[str] = []
        elif op in ("call_module", "call_function", "call_method"):
            # operation node: record the op kind and dependencies
            # For call_module target is typically a string path to a submodule
            # For call_function target is a callable
            op_type = "operation"
            inputs = []
            for arg in node.args:
                # args may be nodes, constants, tuples, etc. Keep node deps only.
                if hasattr(arg, "name"):
                    inputs.append(arg.name)
                elif isinstance(arg, (list, tuple)):
                    for a in arg:
                        if hasattr(a, "name"):
                            inputs.append(a.name)
        elif op == "output":
            op_type = "output"
            # output.args is often a single tuple/list containing node refs
            inputs = []
            for arg in node.args:
                if hasattr(arg, "name"):
                    inputs.append(arg.name)
                elif isinstance(arg, (list, tuple)):
                    for a in arg:
                        if hasattr(a, "name"):
                            inputs.append(a.name)
        else:
            # conservative default mapping
            op_type = str(op)
            inputs = []
            for arg in node.args:
                if hasattr(arg, "name"):
                    inputs.append(arg.name)

        shape = None
        try:
            # FX nodes may carry meta information with tensor metadata
            shape = _extract_shape_from_meta(getattr(node, "meta", {}))
        except Exception:
            shape = None

        # Extract module parameters when lowering from a GraphModule
        node_meta = dict(getattr(node, "meta", {}) or {})
        if graph_module is not None and node.op == "call_module":
            # node.target is a string path to the submodule
            try:
                target = node.target
                # Get submodule safely
                submod = None
                if hasattr(graph_module, "get_submodule"):
                    try:
                        submod = graph_module.get_submodule(target)
                    except Exception:
                        submod = None
                if submod is None:
                    # fallback: traverse attributes
                    parts = str(target).split(".")
                    sub = graph_module
                    for p in parts:
                        sub = getattr(sub, p)
                    submod = sub

                import torch

                # Handle common module types conservatively
                if isinstance(submod, torch.nn.Linear):
                    with torch.no_grad():
                        w = submod.weight.detach().cpu().numpy()
                        node_meta["weight"] = w
                        node_meta["weight_shape"] = tuple(w.shape)
                        if submod.bias is not None:
                            b = submod.bias.detach().cpu().numpy()
                            node_meta["bias"] = b
                            node_meta["bias_shape"] = tuple(b.shape)
                        # Store dtype as numpy dtype name for downstream backends
                        try:
                            node_meta["dtype"] = str(w.dtype.name)
                        except Exception:
                            node_meta["dtype"] = str(getattr(w, "dtype", None))
            except Exception:
                # If param extraction fails, continue gracefully
                pass

        # Preserve FX-level operator identity into meta for later passes
        # Ensure fx_op/target/module_type/source_kind are present in meta
        fx_meta = getattr(node, "meta", {}) or {}
        for k in ("fx_op", "target", "module_type", "source_kind"):
            if k in fx_meta and k not in node_meta:
                node_meta[k] = fx_meta[k]

        ir_node = GraphIRNode(name=name, op_type=op_type, inputs=inputs, shape=shape, meta=node_meta)
        ir.add_node(ir_node)

    return ir


def lower_graph_ir_to_tensor_ir(graph_ir: GraphIR):
    """Lower a GraphIR to a simple TensorIR.

    This converts each operation node into a `TensorOp` preserving order.
    Input and output nodes are skipped; operation nodes become TensorOps.
    """
    try:
        from ir.tensor_ir import TensorIR, TensorOp
    except Exception:  # pragma: no cover - import path issues
        # local import fallback
        from . import tensor_ir as _t
        TensorIR = _t.TensorIR
        TensorOp = _t.TensorOp

    tir = TensorIR()

    for node in graph_ir:
        # Skip graph inputs and outputs
        if node.op_type == "input" or node.op_type == "output":
            continue

        # Map GraphIRNode -> TensorOp
        name = node.name
        op_type = node.op_type
        inputs = list(node.inputs)
        output = node.name
        shape = node.shape
        # Copy node meta -> attributes. If meta contains numpy arrays (weights), keep them.
        # Copy node meta -> attributes. Preserve numpy arrays (weights/bias)
        attributes = dict(node.meta or {})

        top = TensorOp(name=name, op_type=op_type, inputs=inputs, output=output, shape=shape, attributes=attributes)
        tir.add_op(top)

    return tir
