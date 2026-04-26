import json

import mega_kernel_qwen36 as mk


def test_decode_tgraph_contains_events_and_launch_modes():
    graph = mk.build_qwen36_decode_tgraph(
        mk.qwen36_27b_config(),
        batch_size=2,
        num_layers=4,
        worker_config=mk.WorkerConfig.for_device(132),
    )
    plan = graph.linearize(worker_config=mk.WorkerConfig.for_device(132), batch_size=2)
    kinds = {task.kind for task in plan.graph.tasks}
    assert "paged_attention" in kinds
    assert "linear_attention" in kinds
    assert "dense_gate_up" in kinds
    assert any(step.launch_mode == "jit" for step in plan.launch_steps)
    assert any(step.launch_mode == "aot" for step in plan.launch_steps)
    assert plan.worker_config.num_workers == 128
    assert plan.worker_config.num_local_schedulers == 16


def test_moe_tgraph_has_router_and_grouped_experts():
    graph = mk.build_qwen36_decode_tgraph(
        mk.qwen36_35b_a3b_config(),
        batch_size=1,
        num_layers=1,
        include_sampling=False,
        worker_config=mk.WorkerConfig.for_device(132),
    )
    kinds = [task.kind for task in graph.tasks]
    assert "moe_router" in kinds
    assert "moe_grouped_experts" in kinds


def test_batch_specialized_plans_use_powers_of_two():
    plans = mk.build_batch_specialized_plans(
        lambda batch: mk.build_qwen36_decode_tgraph(
            mk.qwen36_27b_config(),
            batch_size=batch,
            num_layers=1,
            worker_config=mk.WorkerConfig.for_device(132),
        ),
        max_batch=8,
        worker_config=mk.WorkerConfig.for_device(132),
    )
    assert tuple(plans) == (1, 2, 4, 8)
    assert all(plan.batch_size == batch for batch, plan in plans.items())


def test_vision_tgraph_and_json_export():
    graph = mk.build_qwen36_vision_tgraph(num_layers=2, sequence_tokens=1024, worker_config=mk.WorkerConfig.for_device(132))
    data = json.loads(graph.to_json())
    assert data["name"].endswith("vision:tokens1024")
    assert any(task["kind"] == "vision_patch_embed" for task in data["tasks"])
    assert any(task["kind"] == "vision_patch_merger" for task in data["tasks"])


def test_mirage_skeleton_mentions_persistent_kernel():
    graph = mk.build_qwen36_decode_tgraph(mk.qwen36_27b_config(), batch_size=1, num_layers=1)
    skeleton = mk.emit_mirage_skeleton(graph.linearize())
    assert "mi.PersistentKernel" in skeleton
    assert "mpk.compile()" in skeleton
