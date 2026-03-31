"""Simple Tensor IR representation.

This module defines a minimal TensorOp dataclass and a TensorIR container
to hold an ordered list of tensor-level operations produced by lowering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class TensorOp:
    name: str
    op_type: str
    inputs: List[str]
    output: str
    shape: Optional[Tuple[int, ...]] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    # Expected attribute fields (convention):
    # - "weight": numpy array for linear weights (canonical layout: (out_features, in_features))
    # - "bias": numpy array for bias (shape: out_features,)
    # - "weight_shape", "bias_shape": shape tuples
    # - "dtype": string/descriptor for dtype
    # Backends may also accept other attributes; attributes may contain
    # numpy arrays, Python scalars, or metadata dicts.


class TensorIR:
    def __init__(self) -> None:
        self.ops: List[TensorOp] = []

    def add_op(self, op: TensorOp) -> None:
        self.ops.append(op)

    def __iter__(self) -> Iterable[TensorOp]:
        return iter(self.ops)

    def __len__(self) -> int:
        return len(self.ops)

    def __repr__(self) -> str:  # pragma: no cover - convenience
        return f"TensorIR(ops={self.ops!r})"
