"""Generate VLM fine-tuning data for document understanding."""
import json, random, argparse
from pathlib import Path
random.seed(42)

DOC_TYPES = ["chart", "table", "report_page", "dashboard_screenshot"]
QUESTIONS = {
    "chart": ["What trend does this chart show?", "What is the peak value?", "Which category is largest?"],
    "table": ["What is the total in the last row?", "Which column has the highest average?", "Extract all values from column 3."],
    "report_page": ["Summarize the key findings.", "What is the main recommendation?", "List all metrics mentioned."],
    "dashboard_screenshot": ["What KPIs are displayed?", "Which metric is in the red zone?", "What time period is shown?"],
}

def main():
    p = argparse.ArgumentParser(); p.add_argument("--n", type=int, default=500)
    p.add_argument("--output_dir", default="data"); a = p.parse_args()
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    examples = []
    for i in range(a.n):
        doc_type = random.choice(DOC_TYPES)
        q = random.choice(QUESTIONS[doc_type])
        examples.append({"image_path": f"images/{doc_type}_{i:04d}.png", "question": q,
                         "doc_type": doc_type, "answer": f"[Placeholder answer for {doc_type} analysis]"})
    split = int(len(examples) * 0.9)
    for name, data in [("train", examples[:split]), ("eval", examples[split:])]:
        with open(out / f"vlm_{name}.jsonl", "w") as f:
            for e in data: f.write(json.dumps(e) + "\n")
    print(f"\u2705 VLM Training Data: {len(examples)} examples")
    for dt in DOC_TYPES:
        n = sum(1 for e in examples if e["doc_type"] == dt)
        print(f"   {dt}: {n}")

if __name__ == "__main__": main()
