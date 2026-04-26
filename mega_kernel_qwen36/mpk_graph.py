from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from .config import Qwen36TextConfig, Qwen36VisionConfig, qwen36_27b_config, qwen36_27b_vision_config


GridDim = tuple[int, int, int]


def _ceil_div(lhs: int, rhs: int) -> int:
    return (lhs + rhs - 1) // rhs


def _pow2_sizes(max_batch: int) -> tuple[int, ...]:
    if max_batch < 1:
        raise ValueError("max_batch must be positive")
    values = []
    size = 1
    while size < max_batch:
        values.append(size)
        size *= 2
    values.append(max_batch)
    return tuple(dict.fromkeys(values))


def _grid_for_dim(dim: int, workers: int, tile: int = 64) -> GridDim:
    blocks = max(1, _ceil_div(dim, tile))
    if workers > 0 and blocks > workers:
        blocks = workers * _ceil_div(blocks, workers)
    return (blocks, 1, 1)


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dims: tuple[int | str, ...]
    dtype: str = "bf16"
    io_category: str = "cuda_tensor"


@dataclass(frozen=True)
class TaskSpec:
    id: str
    kind: str
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    grid_dim: GridDim = (1, 1, 1)
    block_dim: GridDim = (128, 1, 1)
    attrs: Mapping[str, Any] = field(default_factory=dict)
    trigger_event: str = ""
    completion_event: str = ""
    launch_mode: str = "aot"
    worker_hint: Optional[int] = None
    scheduler_hint: Optional[int] = None


@dataclass(frozen=True)
class EventSpec:
    id: str
    prerequisites: tuple[str, ...] = ()
    dependents: tuple[str, ...] = ()
    fused_from: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkerConfig:
    total_sms: int = 132
    scheduler_sms: int = 4
    scheduler_warps_per_sm: int = 4
    shared_memory_page_kb: int = 32

    @property
    def num_workers(self) -> int:
        return max(1, self.total_sms - self.scheduler_sms)

    @property
    def num_local_schedulers(self) -> int:
        return max(1, self.scheduler_sms * self.scheduler_warps_per_sm)

    @classmethod
    def for_device(cls, total_sms: int, *, scheduler_sms: int = 4) -> "WorkerConfig":
        return cls(total_sms=total_sms, scheduler_sms=min(scheduler_sms, max(1, total_sms - 1)))

    @classmethod
    def current_cuda(cls) -> "WorkerConfig":
        try:
            import torch

            if torch.cuda.is_available():
                return cls.for_device(torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count)
        except Exception:
            pass
        return cls()


@dataclass(frozen=True)
class LaunchStep:
    task_id: str
    event_id: str
    launch_mode: str
    worker: Optional[int]
    scheduler: int


@dataclass(frozen=True)
class CompiledPlan:
    graph: "TaskGraph"
    worker_config: WorkerConfig
    launch_steps: tuple[LaunchStep, ...]
    batch_size: int

    def to_dict(self) -> dict[str, Any]:
        data = self.graph.to_dict()
        data["worker_config"] = {
            "total_sms": self.worker_config.total_sms,
            "num_workers": self.worker_config.num_workers,
            "scheduler_sms": self.worker_config.scheduler_sms,
            "num_local_schedulers": self.worker_config.num_local_schedulers,
            "shared_memory_page_kb": self.worker_config.shared_memory_page_kb,
        }
        data["batch_size"] = self.batch_size
        data["launch_steps"] = [step.__dict__ for step in self.launch_steps]
        return data


@dataclass
class TaskGraph:
    name: str
    tensors: dict[str, TensorSpec] = field(default_factory=dict)
    tasks: list[TaskSpec] = field(default_factory=list)
    events: dict[str, EventSpec] = field(default_factory=dict)
    start_event: str = "e_start"

    def add_tensor(
        self,
        name: str,
        dims: Sequence[int | str],
        *,
        dtype: str = "bf16",
        io_category: str = "cuda_tensor",
    ) -> TensorSpec:
        spec = TensorSpec(name=name, dims=tuple(dims), dtype=dtype, io_category=io_category)
        self.tensors[name] = spec
        return spec

    def add_task(
        self,
        task_id: str,
        kind: str,
        *,
        inputs: Iterable[str] = (),
        outputs: Iterable[str] = (),
        depends_on: Iterable[str] = (),
        grid_dim: GridDim = (1, 1, 1),
        block_dim: GridDim = (128, 1, 1),
        attrs: Optional[Mapping[str, Any]] = None,
    ) -> TaskSpec:
        if any(task.id == task_id for task in self.tasks):
            raise ValueError(f"duplicate task id: {task_id}")
        task = TaskSpec(
            id=task_id,
            kind=kind,
            inputs=tuple(inputs),
            outputs=tuple(outputs),
            depends_on=tuple(depends_on),
            grid_dim=grid_dim,
            block_dim=block_dim,
            attrs=dict(attrs or {}),
        )
        self.tasks.append(task)
        return task

    def normalize(self) -> "TaskGraph":
        """Canonicalize task dependencies into MPK-style activation events."""

        events_by_deps: dict[tuple[str, ...], str] = {(): self.start_event}
        dependents: dict[str, list[str]] = {self.start_event: []}
        normalized: list[TaskSpec] = []

        for task in self.tasks:
            deps = tuple(dict.fromkeys(task.depends_on))
            event_id = events_by_deps.get(deps)
            if event_id is None:
                event_id = f"e_{len(events_by_deps)}"
                events_by_deps[deps] = event_id
                dependents[event_id] = []
            dependents.setdefault(event_id, []).append(task.id)
            completion_event = f"e_done_{task.id}"
            normalized.append(replace(task, depends_on=deps, trigger_event=event_id, completion_event=completion_event))

        events: dict[str, EventSpec] = {}
        for deps, event_id in events_by_deps.items():
            fused_from = tuple(f"e_after_{dep}" for dep in deps)
            events[event_id] = EventSpec(
                id=event_id,
                prerequisites=deps,
                dependents=tuple(dependents.get(event_id, ())),
                fused_from=fused_from if len(fused_from) > 1 else (),
            )
        self.tasks = normalized
        self.events = events
        return self

    def topological_tasks(self) -> tuple[TaskSpec, ...]:
        tasks = {task.id: task for task in self.tasks}
        pending = {task.id: set(task.depends_on) for task in self.tasks}
        ordered: list[TaskSpec] = []
        ready = sorted(task_id for task_id, deps in pending.items() if not deps)
        while ready:
            task_id = ready.pop(0)
            ordered.append(tasks[task_id])
            for other_id, deps in pending.items():
                if task_id in deps:
                    deps.remove(task_id)
                    if not deps and other_id not in {task.id for task in ordered} and other_id not in ready:
                        ready.append(other_id)
            ready.sort()
        if len(ordered) != len(self.tasks):
            unresolved = sorted(set(tasks) - {task.id for task in ordered})
            raise ValueError(f"cycle or missing task dependency: {unresolved}")
        return tuple(ordered)

    def event_fusion(self) -> "TaskGraph":
        """Fuse equivalent activation events after graph construction."""

        return self.normalize()

    def linearize(self, worker_config: Optional[WorkerConfig] = None, *, batch_size: int = 1) -> CompiledPlan:
        worker_config = worker_config or WorkerConfig.current_cuda()
        self.event_fusion()
        steps: list[LaunchStep] = []
        deterministic_kinds = {
            "metadata",
            "embed",
            "rmsnorm",
            "linear",
            "rmsnorm_linear",
            "linear_residual",
            "dense_gate_up",
            "dense_down",
            "vision_patch_embed",
            "vision_pos_embed",
            "vision_patch_merger",
            "argmax",
        }
        variable_kinds = {"paged_attention", "split_kv_attention", "linear_attention", "moe_router", "moe_grouped_experts"}
        planned_tasks: list[TaskSpec] = []

        for ordinal, task in enumerate(self.topological_tasks()):
            if task.kind in variable_kinds:
                launch_mode = "jit"
                worker = None
            elif task.kind in deterministic_kinds or len(task.depends_on) == 1:
                launch_mode = "aot"
                worker = ordinal % worker_config.num_workers
            else:
                launch_mode = "jit"
                worker = None
            scheduler = ordinal % worker_config.num_local_schedulers
            planned_tasks.append(replace(task, launch_mode=launch_mode, worker_hint=worker, scheduler_hint=scheduler))
            steps.append(
                LaunchStep(
                    task_id=task.id,
                    event_id=task.trigger_event or self.start_event,
                    launch_mode=launch_mode,
                    worker=worker,
                    scheduler=scheduler,
                )
            )
        self.tasks = planned_tasks
        return CompiledPlan(graph=self, worker_config=worker_config, launch_steps=tuple(steps), batch_size=batch_size)

    def to_dict(self) -> dict[str, Any]:
        self.event_fusion()
        return {
            "name": self.name,
            "start_event": self.start_event,
            "tensors": [tensor.__dict__ for tensor in self.tensors.values()],
            "events": [event.__dict__ for event in self.events.values()],
            "tasks": [
                {
                    **task.__dict__,
                    "attrs": dict(task.attrs),
                }
                for task in self.tasks
            ],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def build_batch_specialized_plans(
    builder: Callable[[int], TaskGraph],
    *,
    max_batch: int,
    worker_config: Optional[WorkerConfig] = None,
) -> dict[int, CompiledPlan]:
    return {
        batch: builder(batch).linearize(worker_config=worker_config, batch_size=batch)
        for batch in _pow2_sizes(max_batch)
    }


def build_qwen36_decode_tgraph(
    config: Optional[Qwen36TextConfig] = None,
    *,
    batch_size: int = 1,
    num_layers: Optional[int] = None,
    include_metadata: bool = True,
    include_sampling: bool = True,
    worker_config: Optional[WorkerConfig] = None,
) -> TaskGraph:
    cfg = config or qwen36_27b_config()
    worker_config = worker_config or WorkerConfig.current_cuda()
    layers = cfg.num_hidden_layers if num_layers is None else min(num_layers, cfg.num_hidden_layers)
    graph = TaskGraph(name=f"{cfg.model_name}:decode:batch{batch_size}")

    graph.add_tensor("tokens", (batch_size, "seq"), dtype="int64")
    graph.add_tensor("hidden_0", (batch_size, cfg.hidden_size))
    if include_metadata:
        graph.add_task(
            "continuous_batching_update",
            "metadata",
            inputs=("tokens",),
            outputs=("decode_metadata",),
            attrs={"paged_attention": True, "select_power_of_two_tgraph": True},
        )
        graph.add_tensor("decode_metadata", (batch_size,), dtype="int32")
        prev = "continuous_batching_update"
    else:
        prev = ""

    graph.add_task(
        "embed",
        "embed",
        inputs=("tokens",),
        outputs=("hidden_0",),
        depends_on=(prev,) if prev else (),
        grid_dim=(1, 1, 1),
        attrs={"vocab_size": cfg.vocab_size},
    )
    prev = "embed"

    for layer_idx in range(layers):
        layer = f"layer_{layer_idx}"
        attn_norm = f"{layer}_attn_norm"
        graph.add_tensor(f"{layer}_attn_norm_out", (batch_size, cfg.hidden_size))
        graph.add_task(
            attn_norm,
            "rmsnorm",
            inputs=(f"hidden_{layer_idx}",),
            outputs=(f"{layer}_attn_norm_out",),
            depends_on=(prev,),
            grid_dim=(batch_size, 1, 1),
        )

        if cfg.layer_types[layer_idx] == "full_attention":
            qkv = f"{layer}_qkv_proj"
            attn = f"{layer}_paged_attention"
            proj = f"{layer}_attn_out_proj_residual"
            q_heads = cfg.num_attention_heads
            kv_heads = cfg.num_key_value_heads
            qkv_dim = (q_heads + 2 * kv_heads) * cfg.head_dim
            graph.add_tensor(f"{layer}_qkv", (batch_size, qkv_dim))
            graph.add_tensor(f"{layer}_attn_out", (batch_size, q_heads * cfg.head_dim))
            graph.add_tensor(f"{layer}_attn_residual", (batch_size, cfg.hidden_size))
            graph.add_task(
                qkv,
                "linear",
                inputs=(f"{layer}_attn_norm_out",),
                outputs=(f"{layer}_qkv",),
                depends_on=(attn_norm,),
                grid_dim=_grid_for_dim(qkv_dim, worker_config.num_workers),
                attrs={"projection": "qkv", "head_dim": cfg.head_dim},
            )
            graph.add_task(
                attn,
                "paged_attention",
                inputs=(f"{layer}_qkv", "decode_metadata"),
                outputs=(f"{layer}_attn_out",),
                depends_on=(qkv,),
                grid_dim=(batch_size, cfg.num_key_value_heads, 1),
                attrs={"causal": True, "kv_heads": kv_heads, "q_heads": q_heads},
            )
            graph.add_task(
                proj,
                "linear_residual",
                inputs=(f"{layer}_attn_out", f"hidden_{layer_idx}"),
                outputs=(f"{layer}_attn_residual",),
                depends_on=(attn,),
                grid_dim=_grid_for_dim(cfg.hidden_size, worker_config.num_workers),
            )
            attn_prev = proj
        else:
            lin = f"{layer}_linear_attention_recurrent"
            graph.add_tensor(f"{layer}_attn_residual", (batch_size, cfg.hidden_size))
            graph.add_task(
                lin,
                "linear_attention",
                inputs=(f"{layer}_attn_norm_out", f"{layer}_linear_state"),
                outputs=(f"{layer}_attn_residual", f"{layer}_linear_state_next"),
                depends_on=(attn_norm,),
                grid_dim=(batch_size, cfg.linear_num_key_heads, 1),
                attrs={
                    "key_heads": cfg.linear_num_key_heads,
                    "value_heads": cfg.linear_num_value_heads,
                    "conv_kernel": cfg.linear_conv_kernel_dim,
                },
            )
            attn_prev = lin

        ffn_norm = f"{layer}_ffn_norm"
        graph.add_tensor(f"{layer}_ffn_norm_out", (batch_size, cfg.hidden_size))
        graph.add_task(
            ffn_norm,
            "rmsnorm",
            inputs=(f"{layer}_attn_residual",),
            outputs=(f"{layer}_ffn_norm_out",),
            depends_on=(attn_prev,),
            grid_dim=(batch_size, 1, 1),
        )

        graph.add_tensor(f"hidden_{layer_idx + 1}", (batch_size, cfg.hidden_size))
        if cfg.is_moe:
            router = f"{layer}_moe_router_topk"
            experts = f"{layer}_moe_grouped_experts"
            output = f"{layer}_moe_down_residual"
            graph.add_tensor(f"{layer}_topk_indices", (batch_size, cfg.num_experts_per_tok), dtype="int64")
            graph.add_tensor(f"{layer}_topk_weights", (batch_size, cfg.num_experts_per_tok), dtype="float32")
            graph.add_tensor(f"{layer}_expert_acts", (batch_size, cfg.num_experts_per_tok, cfg.moe_intermediate_size))
            graph.add_task(
                router,
                "moe_router",
                inputs=(f"{layer}_ffn_norm_out",),
                outputs=(f"{layer}_topk_indices", f"{layer}_topk_weights"),
                depends_on=(ffn_norm,),
                grid_dim=(batch_size, cfg.num_experts, 1),
                attrs={"experts": cfg.num_experts, "top_k": cfg.num_experts_per_tok},
            )
            graph.add_task(
                experts,
                "moe_grouped_experts",
                inputs=(f"{layer}_ffn_norm_out", f"{layer}_topk_indices", f"{layer}_topk_weights"),
                outputs=(f"{layer}_expert_acts",),
                depends_on=(router,),
                grid_dim=(batch_size * cfg.num_experts_per_tok, 1, 1),
                attrs={"intermediate": cfg.moe_intermediate_size, "shared_expert": True},
            )
            graph.add_task(
                output,
                "linear_residual",
                inputs=(f"{layer}_expert_acts", f"{layer}_attn_residual"),
                outputs=(f"hidden_{layer_idx + 1}",),
                depends_on=(experts,),
                grid_dim=_grid_for_dim(cfg.hidden_size, worker_config.num_workers),
            )
            prev = output
        else:
            gate_up = f"{layer}_dense_gate_up"
            down = f"{layer}_dense_down_residual"
            graph.add_tensor(f"{layer}_dense_mid", (batch_size, cfg.intermediate_size))
            graph.add_task(
                gate_up,
                "dense_gate_up",
                inputs=(f"{layer}_ffn_norm_out",),
                outputs=(f"{layer}_dense_mid",),
                depends_on=(ffn_norm,),
                grid_dim=_grid_for_dim(2 * cfg.intermediate_size, worker_config.num_workers),
                attrs={"activation": "silu_mul"},
            )
            graph.add_task(
                down,
                "dense_down",
                inputs=(f"{layer}_dense_mid", f"{layer}_attn_residual"),
                outputs=(f"hidden_{layer_idx + 1}",),
                depends_on=(gate_up,),
                grid_dim=_grid_for_dim(cfg.hidden_size, worker_config.num_workers),
            )
            prev = down

    graph.add_tensor("norm_out", (batch_size, cfg.hidden_size))
    graph.add_task(
        "final_norm",
        "rmsnorm",
        inputs=(f"hidden_{layers}",),
        outputs=("norm_out",),
        depends_on=(prev,),
        grid_dim=(batch_size, 1, 1),
    )
    if include_sampling:
        graph.add_tensor("logits", (batch_size, cfg.vocab_size))
        graph.add_tensor("next_tokens", (batch_size, 1), dtype="int64")
        graph.add_task(
            "lm_head",
            "linear",
            inputs=("norm_out",),
            outputs=("logits",),
            depends_on=("final_norm",),
            grid_dim=_grid_for_dim(cfg.vocab_size, worker_config.num_workers, tile=128),
        )
        graph.add_task(
            "argmax_or_sampler",
            "argmax",
            inputs=("logits",),
            outputs=("next_tokens",),
            depends_on=("lm_head",),
            grid_dim=(worker_config.num_workers, 1, 1),
        )

    return graph.event_fusion()


def build_qwen36_vision_tgraph(
    config: Optional[Qwen36VisionConfig] = None,
    *,
    sequence_tokens: int = 1024,
    num_layers: Optional[int] = None,
    worker_config: Optional[WorkerConfig] = None,
) -> TaskGraph:
    cfg = config or qwen36_27b_vision_config()
    worker_config = worker_config or WorkerConfig.current_cuda()
    layers = cfg.depth if num_layers is None else min(num_layers, cfg.depth)
    merged_tokens = max(1, sequence_tokens // (cfg.spatial_merge_size**2))
    graph = TaskGraph(name=f"{cfg.model_name}:vision:tokens{sequence_tokens}")
    graph.add_tensor("pixel_values", (sequence_tokens, cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size**2))
    graph.add_tensor("vision_hidden_0", (sequence_tokens, cfg.hidden_size))
    graph.add_task(
        "vision_patch_embed",
        "vision_patch_embed",
        inputs=("pixel_values",),
        outputs=("vision_hidden_0",),
        grid_dim=(sequence_tokens, 1, 1),
        attrs={"patch_size": cfg.patch_size, "temporal_patch_size": cfg.temporal_patch_size},
    )
    graph.add_task(
        "vision_pos_embed",
        "vision_pos_embed",
        inputs=("vision_hidden_0",),
        outputs=("vision_hidden_0_pos",),
        depends_on=("vision_patch_embed",),
        grid_dim=(sequence_tokens, 1, 1),
        attrs={"spatial_merge_size": cfg.spatial_merge_size},
    )
    prev = "vision_pos_embed"
    current_hidden = "vision_hidden_0_pos"
    for layer_idx in range(layers):
        layer = f"vision_layer_{layer_idx}"
        graph.add_tensor(f"{layer}_qkv", (sequence_tokens, 3 * cfg.hidden_size))
        graph.add_tensor(f"vision_hidden_{layer_idx + 1}", (sequence_tokens, cfg.hidden_size))
        graph.add_task(
            f"{layer}_ln_qkv",
            "rmsnorm_linear",
            inputs=(current_hidden,),
            outputs=(f"{layer}_qkv",),
            depends_on=(prev,),
            grid_dim=_grid_for_dim(3 * cfg.hidden_size, worker_config.num_workers),
            attrs={"norm": "layernorm", "rotary": True},
        )
        graph.add_task(
            f"{layer}_attention",
            "paged_attention",
            inputs=(f"{layer}_qkv",),
            outputs=(f"{layer}_attn_out",),
            depends_on=(f"{layer}_ln_qkv",),
            grid_dim=(1, cfg.num_heads, 1),
            attrs={"causal": False, "heads": cfg.num_heads, "head_dim": cfg.head_dim},
        )
        graph.add_task(
            f"{layer}_proj_mlp_residual",
            "vision_block_tail",
            inputs=(f"{layer}_attn_out", current_hidden),
            outputs=(f"vision_hidden_{layer_idx + 1}",),
            depends_on=(f"{layer}_attention",),
            grid_dim=_grid_for_dim(max(cfg.hidden_size, cfg.intermediate_size), worker_config.num_workers),
            attrs={"mlp": "gelu_pytorch_tanh", "intermediate": cfg.intermediate_size},
        )
        prev = f"{layer}_proj_mlp_residual"
        current_hidden = f"vision_hidden_{layer_idx + 1}"
    graph.add_tensor("vision_encoded", (merged_tokens, cfg.out_hidden_size))
    graph.add_task(
        "vision_patch_merger",
        "vision_patch_merger",
        inputs=(f"vision_hidden_{layers}",),
        outputs=("vision_encoded",),
        depends_on=(prev,),
        grid_dim=_grid_for_dim(cfg.out_hidden_size, worker_config.num_workers),
        attrs={"merged_hidden": cfg.merged_hidden_size},
    )
    return graph.event_fusion()
