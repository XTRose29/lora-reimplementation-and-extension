from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt_float(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return str(value)


def render_summary_table(rows: List[Dict]) -> str:
    headers = [
        "run",
        "task",
        "method",
        "prompt_variant",
        "bleu",
        "rouge_l",
        "parse_rate",
        "mean_confidence",
        "ece",
        "fact_recall_mean",
    ]
    html_rows = [
        "<tr>" + "".join(f"<th>{html.escape(col)}</th>" for col in headers) + "</tr>"
    ]
    for row in rows:
        html_rows.append(
            "<tr>"
            + "".join(f"<td>{html.escape(str(row.get(col, '')))}</td>" for col in headers)
            + "</tr>"
        )
    return "<table>" + "".join(html_rows) + "</table>"


def render_prediction_card(pred: Dict, idx: int) -> str:
    flags = []
    if pred.get("parse_success"):
        flags.append('<span class="flag parseok">parse-ok</span>')
    else:
        flags.append('<span class="flag parsebad">parse-fail</span>')
    if pred.get("is_correct"):
        flags.append('<span class="flag correct">exact-match</span>')
    if pred.get("wrong_but_high_confidence"):
        flags.append('<span class="flag highwrong">high-conf wrong</span>')
    if pred.get("abstained_by_text"):
        flags.append('<span class="flag abstain">abstained</span>')

    refs = pred.get("references", [])
    refs_json = json.dumps(refs, ensure_ascii=False, indent=2)

    return f"""
    <div class="card">
      <div class="meta">#{idx + 1} {' '.join(flags)}</div>
      <div><strong>Structured Input</strong></div>
      <pre>{html.escape(str(pred.get("structured_input", "")))}</pre>
      <div class="grid2">
        <div>
          <div><strong>Target</strong></div>
          <pre>{html.escape(str(pred.get("target", "")))}</pre>
        </div>
        <div>
          <div><strong>References</strong></div>
          <pre>{html.escape(refs_json)}</pre>
        </div>
      </div>
      <div><strong>Raw Response</strong></div>
      <pre>{html.escape(str(pred.get("raw_response", "")))}</pre>
      <div class="grid2">
        <div>
          <div><strong>Parsed Answer</strong></div>
          <pre>{html.escape(str(pred.get("parsed_answer", "")))}</pre>
        </div>
        <div>
          <div><strong>Confidence</strong></div>
          <pre>{html.escape(str(pred.get("confidence_int", "")))} ({fmt_float(pred.get("confidence"))})</pre>
        </div>
      </div>
      <div class="grid4">
        <div><strong>ROUGE-L</strong><pre>{fmt_float(pred.get("reference_rouge_l"))}</pre></div>
        <div><strong>Fact Recall</strong><pre>{fmt_float(pred.get("fact_recall"))}</pre></div>
        <div><strong>Answer Tokens</strong><pre>{html.escape(str(pred.get("answer_length_tokens", "")))}</pre></div>
        <div><strong>Response Tokens</strong><pre>{html.escape(str(pred.get("response_length_tokens", "")))}</pre></div>
      </div>
      <div class="grid4">
        <div><strong>Seq LogProb</strong><pre>{fmt_float(pred.get("sequence_logprob_mean"))}</pre></div>
        <div><strong>Top1-Top2 Margin</strong><pre>{fmt_float(pred.get("top1_top2_margin_mean"))}</pre></div>
        <div><strong>Entropy</strong><pre>{fmt_float(pred.get("entropy_mean"))}</pre></div>
        <div><strong>Repetition 3-gram</strong><pre>{fmt_float(pred.get("repetition_ngram_rate"))}</pre></div>
      </div>
    </div>
    """


def sort_predictions(rows: List[Dict]) -> List[Dict]:
    def key_fn(pred: Dict):
        return (
            0 if pred.get("wrong_but_high_confidence") else 1,
            0 if not pred.get("parse_success") else 1,
            -float(pred.get("confidence") or 0.0),
        )

    return sorted(rows, key=key_fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HTML for NLG smoke outputs.")
    parser.add_argument("--results_root", required=True)
    parser.add_argument("--output_html", required=True)
    parser.add_argument("--samples_per_run", type=int, default=10)
    args = parser.parse_args()

    results_root = Path(args.results_root)
    output_html = Path(args.output_html)
    run_dirs = sorted(path for path in results_root.rglob("*") if path.is_dir())

    sections: List[str] = []
    summary_rows: List[Dict] = []
    tab_buttons: List[str] = []
    first_tab_id: str | None = None
    for run_dir in run_dirs:
        metrics_path = run_dir / "metrics.json"
        config_path = run_dir / "train_config.json"
        pred_path = run_dir / "id_predictions.jsonl"
        if not (metrics_path.exists() and config_path.exists() and pred_path.exists()):
            continue

        metrics = load_json(metrics_path).get("final_id", {})
        config = load_json(config_path)
        preds = sort_predictions(load_jsonl(pred_path))[: args.samples_per_run]

        summary_rows.append(
            {
                "run": run_dir.name,
                "task": config.get("dataset_task"),
                "method": config.get("method"),
                "prompt_variant": config.get("prompt_variant"),
                "bleu": fmt_float(metrics.get("bleu")),
                "rouge_l": fmt_float(metrics.get("rouge_l")),
                "parse_rate": fmt_float(metrics.get("parse_rate")),
                "mean_confidence": fmt_float(metrics.get("mean_confidence")),
                "ece": fmt_float(metrics.get("ece")),
                "fact_recall_mean": fmt_float(metrics.get("fact_recall_mean")),
            }
        )

        cards = "\n".join(render_prediction_card(pred, idx) for idx, pred in enumerate(preds))
        tab_id = f"tab_{run_dir.name}"
        if first_tab_id is None:
            first_tab_id = tab_id
        label = f"{config.get('dataset_task')} | {config.get('prompt_variant')} | {config.get('method')}"
        tab_buttons.append(
            f'<button class="tablink" data-target="{html.escape(tab_id)}">{html.escape(str(label))}</button>'
        )
        sections.append(
            f"""
            <section class="tabcontent" id="{html.escape(tab_id)}">
              <h2>{html.escape(run_dir.name)}</h2>
              <p class="sub">
                task={html.escape(str(config.get("dataset_task")))} |
                method={html.escape(str(config.get("method")))} |
                prompt={html.escape(str(config.get("prompt_variant")))} |
                bleu={fmt_float(metrics.get("bleu"))} |
                parse_rate={fmt_float(metrics.get("parse_rate"))} |
                mean_conf={fmt_float(metrics.get("mean_confidence"))}
              </p>
              {cards}
            </section>
            """
        )

    output_html.parent.mkdir(parents=True, exist_ok=True)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>NLG Smoke Samples</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      line-height: 1.4;
      color: #111;
      background: #fafafa;
    }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .sub {{ color: #444; margin: 0 0 16px 0; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-bottom: 28px;
      background: white;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 8px 10px;
      text-align: left;
      font-size: 13px;
    }}
    th {{ background: #f0f0f0; }}
    section {{ margin-bottom: 40px; }}
    .tabbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin: 0 0 20px 0;
    }}
    .tablink {{
      border: 1px solid #bbb;
      background: white;
      border-radius: 6px;
      padding: 8px 10px;
      cursor: pointer;
      font-size: 13px;
    }}
    .tablink.active {{
      background: #111;
      color: white;
      border-color: #111;
    }}
    .tabcontent {{
      display: none;
    }}
    .tabcontent.active {{
      display: block;
    }}
    .card {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 6px;
      padding: 12px;
      margin-bottom: 12px;
    }}
    .meta {{
      margin-bottom: 8px;
      font-size: 12px;
      color: #333;
    }}
    .flag {{
      display: inline-block;
      margin-right: 8px;
      padding: 2px 6px;
      border-radius: 4px;
      font-size: 12px;
      color: white;
    }}
    .correct {{ background: #2e7d32; }}
    .abstain {{ background: #6a1b9a; }}
    .highwrong {{ background: #ef6c00; }}
    .parseok {{ background: #1565c0; }}
    .parsebad {{ background: #c62828; }}
    .grid2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .grid4 {{
      display: grid;
      grid-template-columns: 1fr 1fr 1fr 1fr;
      gap: 12px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #f7f7f7;
      padding: 8px;
      border-radius: 4px;
      margin: 4px 0 8px 0;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <h1>NLG Smoke Samples</h1>
  <p class="sub">Each run shows up to {args.samples_per_run} examples, sorted with high-confidence wrong and parse failures first.</p>
  {render_summary_table(summary_rows)}
  <div class="tabbar">
    {''.join(tab_buttons)}
  </div>
  {''.join(sections)}
  <script>
    const buttons = Array.from(document.querySelectorAll('.tablink'));
    const contents = Array.from(document.querySelectorAll('.tabcontent'));
    function activateTab(tabId) {{
      contents.forEach((el) => el.classList.toggle('active', el.id === tabId));
      buttons.forEach((btn) => btn.classList.toggle('active', btn.dataset.target === tabId));
    }}
    buttons.forEach((btn) => {{
      btn.addEventListener('click', () => activateTab(btn.dataset.target));
    }});
    {"activateTab('" + first_tab_id + "');" if first_tab_id else ""}
  </script>
</body>
</html>
"""
    output_html.write_text(html_text, encoding="utf-8")
    print(f"Wrote {output_html}")


if __name__ == "__main__":
    main()
