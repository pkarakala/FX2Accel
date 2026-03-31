import torch

from models.mlp import SimpleMLP
from frontend.fx_capture import capture_with_shape
from lowering.graph_to_tensor_ir import lower_fx_to_graph_ir


def test_fx_shape_propagation_and_node_order():
    model = SimpleMLP(in_features=4, hidden=4, out_features=4)
    example_input = torch.randn(1, 4)

    gm = capture_with_shape(model, example_input)
    ir = lower_fx_to_graph_ir(gm)

    # At least one node should have shape metadata populated
    shapes = [n.shape for n in ir if n.shape is not None]
    assert len(shapes) >= 1

    # Expect node op_type ordering: input -> operation -> operation -> operation -> output
    types = [n.op_type for n in ir]
    # We expect at least input, operation, output in sequence
    assert types[0] == "input"
    assert types[-1] == "output"


def test_dead_code_elimination_removes_unused_nodes():
    # Construct a simple GraphIR manually
    from lowering.graph_to_tensor_ir import GraphIR, GraphIRNode
    from passes.dead_code_elimination import run_dead_code_elimination

    ir = GraphIR()
    ir.add_node(GraphIRNode("x", "input", inputs=[]))
    ir.add_node(GraphIRNode("a", "operation", inputs=["x"]))
    ir.add_node(GraphIRNode("b", "operation", inputs=["a"]))
    ir.add_node(GraphIRNode("unused", "operation", inputs=["x"]))
    ir.add_node(GraphIRNode("out", "output", inputs=["b"]))

    new_ir = run_dead_code_elimination(ir)
    names = [n.name for n in new_ir]
    assert "unused" not in names
    assert "x" in names and "out" in names


def test_op_fusion_merges_linear_relu():
    from lowering.graph_to_tensor_ir import GraphIR, GraphIRNode
    from passes.op_fusion import run_op_fusion

    ir = GraphIR()
    ir.add_node(GraphIRNode("x", "input", inputs=[]))
    # mark producer as linear-like via meta
    ir.add_node(GraphIRNode("lin", "operation", inputs=["x"], meta={"module_type": "Linear"}))
    # mark consumer as relu via meta
    ir.add_node(GraphIRNode("relu", "operation", inputs=["lin"], meta={"module_type": "ReLU"}))
    ir.add_node(GraphIRNode("out", "output", inputs=["relu"]))

    new_ir = run_op_fusion(ir)
    types = [n.op_type for n in new_ir]
    # Expect a fused op_type present
    assert "linear_relu" in types
    # Ensure output now consumes the fused node
    out_node = [n for n in new_ir if n.op_type == "output"][0]
    assert len(out_node.inputs) == 1
    assert out_node.inputs[0] != "relu"


def test_fusion_removes_intermediate_and_weight_layout():
    import torch
    import numpy as np

    from frontend.fx_capture import capture_with_shape
    from lowering.graph_to_tensor_ir import lower_fx_to_graph_ir, lower_graph_ir_to_tensor_ir
    from passes.op_fusion import run_op_fusion
    from passes.dead_code_elimination import run_dead_code_elimination

    model = SimpleMLP(in_features=4, hidden=4, out_features=4)
    example_input = torch.randn(1, 4)
    gm = capture_with_shape(model, example_input)

    gir = lower_fx_to_graph_ir(gm)
    gir = run_dead_code_elimination(gir)
    gir = run_op_fusion(gir)

    # Ensure that the original 'lin' node is not present if fused
    names = [n.name for n in gir]
    assert not any(n.endswith("lin") and n == "lin" for n in names) or "linear_relu" in [n.op_type for n in gir]

    tir = lower_graph_ir_to_tensor_ir(gir)
    # Check that linear/linear_relu ops have canonical weight layout if present
    for op in tir:
        if op.op_type in ("operation", "linear_relu"):
            w = op.attributes.get("weight")
            if w is not None:
                # canonical layout: (out_features, in_features)
                assert isinstance(w, np.ndarray)
                assert w.ndim == 2


def test_fusion_uses_metadata_not_names():
    # Build a minimal GraphIR with odd names but correct metadata
    from lowering.graph_to_tensor_ir import GraphIR, GraphIRNode
    from passes.op_fusion import run_op_fusion

    gir = GraphIR()
    gir.add_node(GraphIRNode("in_abc", "input", inputs=[], meta={}))
    # producer with non-linear name but linear metadata
    gir.add_node(GraphIRNode("node_42", "operation", inputs=["in_abc"], meta={"module_type": "Linear", "weight": None}))
    # relu consumer with odd name but relu metadata
    gir.add_node(GraphIRNode("weird_relu_name", "operation", inputs=["node_42"], meta={"module_type": "ReLU"}))
    gir.add_node(GraphIRNode("out_z", "output", inputs=["weird_relu_name"], meta={}))

    new_gir = run_op_fusion(gir)
    # Expect a fused node present and original producer/consumer removed
    names = [n.name for n in new_gir]
    assert any(n for n in names if "fused" in n)
    assert "node_42" not in names and "weird_relu_name" not in names


def test_lower_graph_ir_to_tensor_ir():
    from lowering.graph_to_tensor_ir import lower_graph_ir_to_tensor_ir
    from ir.tensor_ir import TensorIR, TensorOp
    from lowering.graph_to_tensor_ir import lower_fx_to_graph_ir
    from frontend.fx_capture import capture_with_shape

    model = SimpleMLP(in_features=4, hidden=4, out_features=4)
    example_input = torch.randn(1, 4)
    gm = capture_with_shape(model, example_input)
    gir = lower_fx_to_graph_ir(gm)

    # Run passes to potentially produce a fused node
    from passes.dead_code_elimination import run_dead_code_elimination
    from passes.op_fusion import run_op_fusion

    gir = run_dead_code_elimination(gir)
    gir = run_op_fusion(gir)

    tir = lower_graph_ir_to_tensor_ir(gir)
    # Expect TensorIR to be non-empty and contain op entries
    assert len(tir) > 0
    types = [op.op_type for op in tir]
    # Expect either operation or linear_relu present
    assert any(t in ("operation", "linear_relu") for t in types)


def test_memory_planner_estimates_and_reuses():
    from ir.tensor_ir import TensorIR, TensorOp
    from memory.planner import plan_memory

    # Build a TensorIR with non-overlapping lifetimes to enable reuse
    tir = TensorIR()
    tir.add_op(TensorOp(name="t0_op", op_type="op", inputs=[], output="t0", shape=(10, 10)))
    tir.add_op(TensorOp(name="t1_op", op_type="op", inputs=["t0"], output="t1", shape=(10, 10)))
    tir.add_op(TensorOp(name="t2_op", op_type="op", inputs=[], output="t2", shape=(10, 10)))
    tir.add_op(TensorOp(name="t3_op", op_type="op", inputs=["t2"], output="t3", shape=(10, 10)))

    plan = plan_memory(tir, element_size_bytes=4)
    assert plan["peak_bytes"] > 0
    # Expect at least one reuse: t2 can reuse t0's buffer
    assert len(plan["reused"]) >= 1


def test_accel_codegen_and_numpy_backend():
    import numpy as np

    from ir.tensor_ir import TensorIR, TensorOp
    from backend.accel_backend import emit_instructions
    from backend.numpy_backend import run_tensor_ir

    # Create a small TensorIR with a linear_relu fusion candidate
    tir = TensorIR()
    tir.add_op(TensorOp(name="lin", op_type="operation", inputs=["x"], output="lin", shape=(1, 4)))
    tir.add_op(TensorOp(name="lin_relu_fused", op_type="linear_relu", inputs=["lin"], output="lin_relu_fused", shape=(1, 4)))

    instrs = emit_instructions(tir)
    assert len(instrs) > 0

    # Run numpy backend: provide input x
    x = np.random.randn(1, 4).astype(np.float32)
    out = run_tensor_ir(tir, {"x": x})
    assert out.shape == (1, 4)


def test_numpy_backend_matches_pytorch():
    import torch
    import numpy as np

    from frontend.fx_capture import capture_with_shape
    from lowering.graph_to_tensor_ir import lower_fx_to_graph_ir
    from lowering.graph_to_tensor_ir import lower_graph_ir_to_tensor_ir
    from passes.dead_code_elimination import run_dead_code_elimination
    from passes.op_fusion import run_op_fusion
    from backend.numpy_backend import run_tensor_ir

    model = SimpleMLP(in_features=4, hidden=4, out_features=4)
    model.eval()
    example_input = torch.randn(1, 4)
    gm = capture_with_shape(model, example_input)

    gir = lower_fx_to_graph_ir(gm)
    gir = run_dead_code_elimination(gir)
    gir = run_op_fusion(gir)
    tir = lower_graph_ir_to_tensor_ir(gir)

    # Build inputs: map 'x' to example_input
    inputs = {"x": example_input.numpy().astype(np.float32)}
    out_np = run_tensor_ir(tir, inputs)

    with torch.no_grad():
        out_pt = model(example_input).cpu().numpy()

    assert out_np.shape == out_pt.shape
    # numerical comparison: allow small tolerance
    diff = float(np.max(np.abs(out_np - out_pt)))
    assert diff < 1e-4
