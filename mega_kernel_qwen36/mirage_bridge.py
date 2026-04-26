from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from .mpk_graph import CompiledPlan, TaskGraph, WorkerConfig


@dataclass(frozen=True)
class MirageKernelConfig:
    world_size: int = 1
    mpi_rank: int = 0
    max_seq_length: int = 262144
    max_num_pages: int = 16
    page_size: int = 4096
    mode: str = "offline"
    use_cutlass_kernel: bool = True


def mirage_available() -> bool:
    try:
        import mirage  # noqa: F401
    except Exception:
        return False
    return True


def persistent_kernel_kwargs(
    worker_config: WorkerConfig,
    *,
    config: Optional[MirageKernelConfig] = None,
    max_num_batched_requests: int = 1,
    max_num_batched_tokens: int = 1,
) -> dict[str, Any]:
    cfg = config or MirageKernelConfig()
    return {
        "mode": cfg.mode,
        "world_size": cfg.world_size,
        "mpi_rank": cfg.mpi_rank,
        "num_workers": worker_config.num_workers,
        "num_local_schedulers": worker_config.num_local_schedulers,
        "num_remote_schedulers": 0,
        "max_seq_length": cfg.max_seq_length,
        "max_num_batched_requests": max_num_batched_requests,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_pages": cfg.max_num_pages,
        "page_size": cfg.page_size,
        "use_cutlass_kernel": cfg.use_cutlass_kernel,
    }


def emit_mirage_skeleton(
    graph_or_plan: TaskGraph | CompiledPlan,
    *,
    config: Optional[MirageKernelConfig] = None,
) -> str:
    """Emit a Mirage MPK-oriented Python skeleton for a local tGraph.

    The generated code is intentionally a bridge artifact: it lists tensors and
    tasks using Mirage's PersistentKernel vocabulary, while unsupported local
    task kinds are left as explicit TODO comments.
    """

    if isinstance(graph_or_plan, CompiledPlan):
        plan = graph_or_plan
        graph = plan.graph
        worker_config = plan.worker_config
        batch_size = plan.batch_size
    else:
        graph = graph_or_plan
        plan = graph.linearize()
        worker_config = plan.worker_config
        batch_size = 1

    kwargs = persistent_kernel_kwargs(
        worker_config,
        config=config,
        max_num_batched_requests=batch_size,
        max_num_batched_tokens=batch_size,
    )
    lines = [
        "import mirage as mi",
        "",
        "# Generated from mega_kernel_qwen36.mpk_graph.",
        "# Fill the torch_tensor bindings with live model tensors before compile().",
        f"mpk = mi.PersistentKernel({', '.join(f'{key}={value!r}' for key, value in kwargs.items())})",
        "",
        "tensors = {}",
    ]

    for tensor in graph.tensors.values():
        dtype = {
            "bf16": "mi.bfloat16",
            "float32": "mi.float32",
            "int32": "mi.int32",
            "int64": "mi.int64",
        }.get(tensor.dtype, "mi.bfloat16")
        if any(isinstance(dim, str) for dim in tensor.dims):
            dims = tuple(1 if isinstance(dim, str) else dim for dim in tensor.dims)
            lines.append(f"# TODO dynamic dims for {tensor.name}: {tensor.dims!r}")
        else:
            dims = tensor.dims
        lines.append(
            f"tensors[{tensor.name!r}] = mpk.new_tensor("
            f"dims={dims!r}, dtype={dtype}, name={tensor.name!r}, io_category={tensor.io_category!r})"
        )

    lines.append("")
    lines.append("# Task graph")
    for step in plan.launch_steps:
        task = next(task for task in graph.tasks if task.id == step.task_id)
        lines.append(f"# {task.id}: kind={task.kind}, launch={step.launch_mode}, event={step.event_id}")
        if task.kind == "rmsnorm":
            lines.append("# mpk.rmsnorm_layer(input=..., weight=..., output=..., grid_dim=..., block_dim=...)")
        elif task.kind == "linear":
            lines.append("# mpk.linear_layer(input=..., weight=..., output=..., grid_dim=..., block_dim=...)")
        elif task.kind == "rmsnorm_linear":
            lines.append("# mpk.rmsnorm_linear_layer(input=..., weight_norm=..., weight_linear=..., output=...)")
        elif task.kind == "linear_residual":
            lines.append("# mpk.linear_with_residual_layer(input=..., weight=..., residual=..., output=...)")
        elif task.kind == "paged_attention":
            lines.append("# mpk.paged_attention_layer(input=..., k_cache=..., v_cache=..., output=...)")
        elif task.kind == "embed":
            lines.append("# mpk.embed_layer(input=..., weight=..., output=...)")
        elif task.kind in {"dense_gate_up", "dense_down", "moe_router", "moe_grouped_experts", "linear_attention"}:
            lines.append(f"# TODO map local task kind {task.kind!r} to Mirage fused/CUTLASS tasks.")
        else:
            lines.append(f"# TODO task kind {task.kind!r}")

    lines.append("")
    lines.append("mpk.compile()")
    lines.append("# mpk()")
    return "\n".join(lines) + "\n"
