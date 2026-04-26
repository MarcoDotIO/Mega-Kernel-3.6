from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import mega_kernel_qwen36 as mk


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an MPK-style tGraph plan for Qwen3.6 kernels.")
    parser.add_argument("--model", choices=["27b", "35b-a3b", "vision"], default="27b")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-batch", type=int)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--total-sms", type=int, default=132)
    parser.add_argument("--scheduler-sms", type=int, default=4)
    parser.add_argument("--mirage-skeleton", action="store_true")
    args = parser.parse_args()

    worker_config = mk.WorkerConfig.for_device(args.total_sms, scheduler_sms=args.scheduler_sms)

    def build(batch: int) -> mk.TaskGraph:
        if args.model == "35b-a3b":
            return mk.build_qwen36_decode_tgraph(
                mk.qwen36_35b_a3b_config(),
                batch_size=batch,
                num_layers=args.layers,
                worker_config=worker_config,
            )
        if args.model == "vision":
            return mk.build_qwen36_vision_tgraph(
                mk.qwen36_27b_vision_config(),
                sequence_tokens=args.tokens,
                num_layers=args.layers,
                worker_config=worker_config,
            )
        return mk.build_qwen36_decode_tgraph(
            mk.qwen36_27b_config(),
            batch_size=batch,
            num_layers=args.layers,
            worker_config=worker_config,
        )

    if args.max_batch is not None:
        plans = mk.build_batch_specialized_plans(build, max_batch=args.max_batch, worker_config=worker_config)
        print({batch: {"tasks": len(plan.graph.tasks), "events": len(plan.graph.events)} for batch, plan in plans.items()})
        return

    plan = build(args.batch_size).linearize(worker_config=worker_config, batch_size=args.batch_size)
    if args.mirage_skeleton:
        print(mk.emit_mirage_skeleton(plan))
    else:
        print(plan.graph.to_json())


if __name__ == "__main__":
    main()
