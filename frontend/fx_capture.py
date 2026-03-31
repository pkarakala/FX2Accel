"""Helpers to capture FX graphs and run shape propagation."""

from __future__ import annotations

from typing import Any


def capture_with_shape(module: Any, example_input: Any):
    """Symbolically trace `module` with FX and run shape propagation.

    Args:
        module: a torch.nn.Module instance
        example_input: example input (tensor or tuple of tensors) matching forward

    Returns:
        A traced GraphModule with populated `node.meta['tensor_meta']` when available.
    """
    try:
        import torch.fx as fx
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("torch.fx is required for FX capture") from e

    gm = fx.symbolic_trace(module)

    # Run shape propagation to populate node.meta with tensor metadata
    try:
        from torch.fx.passes.shape_prop import ShapeProp

        sp = ShapeProp(gm)
        # `propagate` accepts the same argument structure as forward
        sp.propagate(example_input)
    except Exception:
        # If shape propagation is unavailable or fails, return unmapped gm
        return gm

    # Preserve useful FX node metadata to aid later lowering and passes.
    for node in gm.graph.nodes:
        # Ensure meta dict exists
        meta = getattr(node, "meta", {}) or {}
        meta["fx_op"] = node.op
        meta["target"] = getattr(node, "target", None)
        meta["source_kind"] = node.op
        # If this is a call_module, try to record the module type
        if node.op == "call_module":
            try:
                # get_submodule may not exist on older FX versions
                submod = gm.get_submodule(node.target) if hasattr(gm, "get_submodule") else None
            except Exception:
                submod = None
            if submod is None:
                # fallback traversal
                try:
                    parts = str(node.target).split(".")
                    sub = gm
                    for p in parts:
                        sub = getattr(sub, p)
                    submod = sub
                except Exception:
                    submod = None

            if submod is not None:
                meta["module_type"] = type(submod).__name__
        # If call_method with 'relu', tag as ReLU
        if node.op == "call_method" and str(node.target) == "relu":
            meta["module_type"] = "ReLU"

        node.meta = meta

    return gm
