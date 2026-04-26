from __future__ import annotations

import json
import urllib.request
import argparse


DEFAULT_MODEL = "Qwen/Qwen3.6-27B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    url = f"https://huggingface.co/{args.model}/raw/main/config.json"
    with urllib.request.urlopen(url, timeout=30) as response:
        config = json.load(response)
    text = config["text_config"]
    summary = {
        "model": args.model,
        "model_type": text["model_type"],
        "hidden_size": text["hidden_size"],
        "num_hidden_layers": text["num_hidden_layers"],
        "layer_types_first_8": text["layer_types"][:8],
        "num_attention_heads": text["num_attention_heads"],
        "num_key_value_heads": text["num_key_value_heads"],
        "head_dim": text["head_dim"],
        "intermediate_size": text.get("intermediate_size"),
        "num_experts": text.get("num_experts"),
        "num_experts_per_tok": text.get("num_experts_per_tok"),
        "moe_intermediate_size": text.get("moe_intermediate_size"),
        "max_position_embeddings": text["max_position_embeddings"],
    }
    vision = config.get("vision_config")
    if vision is not None:
        summary["vision"] = {
            "depth": vision["depth"],
            "hidden_size": vision["hidden_size"],
            "intermediate_size": vision["intermediate_size"],
            "num_heads": vision["num_heads"],
            "patch_size": vision["patch_size"],
            "temporal_patch_size": vision["temporal_patch_size"],
            "spatial_merge_size": vision["spatial_merge_size"],
            "out_hidden_size": vision["out_hidden_size"],
            "num_position_embeddings": vision["num_position_embeddings"],
            "deepstack_visual_indexes": vision.get("deepstack_visual_indexes", []),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
