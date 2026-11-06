"""Section "Tool retrieval is RAG over actions": context cost of a catalogue, and recall as a ceiling on success.

    python labs/the-making-of-an-ai-agent/scripts/06_tool_retrieval.py
Schema tokens are counted with the SmolLM2-135M tokenizer if it is in the local Hugging Face cache; otherwise the
script says so and counts regex word pieces instead.
"""

import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentcraft.common import table, write_results
from agentcraft.tool_retrieval import TfidfIndex, catalogue, evaluate

SIZES, KS, TASKS = [100, 1000, 10000], [3, 8, 16, 32, 64], 500


def schema_tokens(schemas: list[str]) -> tuple[float, str]:
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M", local_files_only=True)
        return statistics.mean(len(tok(s)["input_ids"]) for s in schemas), "SmolLM2-135M tokenizer"
    except Exception:  # noqa: BLE001 - no cached tokenizer: fall back, and say so
        return statistics.mean(len(re.findall(r"\w+|[^\w\s]", s)) for s in schemas), "regex word pieces"


def main():
    sample = random.Random(0).sample(catalogue(10000), 300)
    s, counter = schema_tokens([t.schema() for t in sample])
    cost_rows = [[n, f"{s * n:,.0f}"] + [f"{s * k:,.0f}" for k in (8, 32)] for n in SIZES + [16464]]
    rows, data = [], []
    for size in SIZES:
        index = TfidfIndex(catalogue(size))
        for k in KS:
            for decomposed in (False, True):
                r = evaluate(index, k, decomposed, TASKS, seed=1)
                data.append({"catalogue": size, "k": k, "decomposed": decomposed, **r})
                rows.append([size, k, "per clause" if decomposed else "whole request", r["per_tool_recall"],
                             r["all_three_retrieved"], r["independence_r3"]])
    text = (f"Mean schema length: {s:.1f} tokens ({counter}, 300 sampled tools).\n\n"
            "Context cost in tokens (16,464 is ToolBench's API count, for scale; its schemas are not used here):\n\n"
            + table(cost_rows, ["catalogue size", "all schemas in prompt", "top-8 in prompt", "top-32 in prompt"])
            + f"\n\nRetrieval over the synthetic catalogue, TF-IDF cosine, {TASKS} three-tool tasks per row "
            "(each word phrased with a synonym half the time).\n'all three retrieved' is the ceiling on task "
            "success: a task whose tools are not in the context cannot be solved.\n\n"
            + table(rows, ["catalogue", "k", "retrieval", "per-tool recall", "all three retrieved (ceiling)",
                           "recall^3 (independence)"]))
    print(text)
    write_results("06_tool_retrieval", text, {"schema_tokens": s, "counter": counter, "rows": data})


if __name__ == "__main__":
    main()
