"""Simple memory planner for TensorIR.

This module performs liveness analysis on TensorIR outputs, estimates
peak memory (assuming float32 tensors), and performs a conservative
buffer reuse assignment when lifetimes do not overlap.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ir.tensor_ir import TensorIR, TensorOp


def _num_elements(shape: Optional[Tuple[int, ...]]) -> int:
    if not shape:
        return 0
    prod = 1
    for d in shape:
        prod *= int(d)
    return prod


def plan_memory(tir: TensorIR, element_size_bytes: int = 4):
    """Plan memory for the given TensorIR.

    Returns a dict with:
      - lifetimes: mapping tensor name -> (start_idx, end_idx)
      - peak_bytes: estimated peak memory in bytes
      - assignments: mapping tensor name -> buffer_id
      - buffers: mapping buffer_id -> size_bytes
      - reused: list of (tensor, reused_from_buffer_id)
    """
    # First pass: determine producer index for each tensor and record input usages
    producer_idx: Dict[str, int] = {}
    last_use: Dict[str, int] = {}

    for idx, op in enumerate(tir.ops):
        # op.output is produced at idx
        producer_idx[op.output] = idx
        # inputs are used at idx
        for inp in op.inputs:
            last_use[inp] = max(last_use.get(inp, -1), idx)

    # Determine lifetimes for produced tensors
    lifetimes: Dict[str, Tuple[int, int]] = {}
    for op in tir.ops:
        name = op.output
        start = producer_idx.get(name, 0)
        end = last_use.get(name, start)
        lifetimes[name] = (start, end)

    # Simulate allocation with buffer reuse
    # free_buffers: list of (buffer_id, size_bytes)
    free_buffers: List[Tuple[int, int]] = []
    assignments: Dict[str, int] = {}
    buffers: Dict[int, int] = {}
    reused: List[Tuple[str, int]] = []
    current_buffers: Dict[int, int] = {}  # buffer_id -> size
    next_buffer_id = 0

    # Map end times to list of tensors to free after that op
    end_buckets: Dict[int, List[str]] = {}
    for name, (s, e) in lifetimes.items():
        end_buckets.setdefault(e, []).append(name)

    peak_bytes = 0

    # Iterate ops in order and allocate buffers for each op's output
    for idx, op in enumerate(tir.ops):
        # Free buffers whose lifetimes ended before this op (end < idx)
        for e in list(end_buckets.keys()):
            if e < idx:
                for name in end_buckets.get(e, []):
                    b = assignments.get(name)
                    if b is not None:
                        size = buffers.get(b, 0)
                        free_buffers.append((b, size))
                        if b in current_buffers:
                            del current_buffers[b]
                del end_buckets[e]

        # Allocate buffer for this op's output
        out_name = op.output
        shape = op.shape
        numel = _num_elements(shape)
        size_bytes = numel * element_size_bytes

        # Try to reuse a free buffer with sufficient size (first-fit)
        chosen_buf: Optional[int] = None
        for i, (buf_id, buf_size) in enumerate(free_buffers):
            if buf_size >= size_bytes:
                chosen_buf = buf_id
                # remove from free list
                free_buffers.pop(i)
                reused.append((out_name, chosen_buf))
                break

        if chosen_buf is None:
            chosen_buf = next_buffer_id
            next_buffer_id += 1
            buffers[chosen_buf] = size_bytes

        assignments[out_name] = chosen_buf
        current_buffers[chosen_buf] = buffers[chosen_buf]

        # Update peak
        current_total = sum(current_buffers.values())
        if current_total > peak_bytes:
            peak_bytes = current_total

        # Note: we will actually free buffers after their end when end < next_idx via end_buckets processing

    return {
        "lifetimes": lifetimes,
        "peak_bytes": peak_bytes,
        "assignments": assignments,
        "buffers": buffers,
        "reused": reused,
    }
