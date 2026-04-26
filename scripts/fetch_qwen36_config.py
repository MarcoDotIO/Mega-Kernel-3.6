from __future__ import annotations

import json
import urllib.request


URL = "https://huggingface.co/Qwen/Qwen3.6-35B-A3B/raw/main/config.json"


def main() -> None:
    with urllib.request.urlopen(URL, timeout=30) as response:
        config = json.load(response)
    text = config["text_config"]
    summary = {
        "model_type": text["model_type"],
        "hidden_size": text["hidden_size"],
        "num_hidden_layers": text["num_hidden_layers"],
        "layer_types_first_8": text["layer_types"][:8],
        "num_attention_heads": text["num_attention_heads"],
        "num_key_value_heads": text["num_key_value_heads"],
        "head_dim": text["head_dim"],
        "num_experts": text["num_experts"],
        "num_experts_per_tok": text["num_experts_per_tok"],
        "moe_intermediate_size": text["moe_intermediate_size"],
        "max_position_embeddings": text["max_position_embeddings"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
