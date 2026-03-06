#!/usr/bin/env python3
"""Compare Gemini models on garage door classification.

Runs the same prompt + images against multiple models and prints
a comparison table with status, confidence, reasoning, tokens, and cost.

Usage:
    export GM_GEMINI_API_KEY=your-key
    python scripts/compare_models.py [image_dir_or_files...]

If no arguments given, looks for test images in test_images/.
"""

import json
import os
import sys
import time

from google import genai
from google.genai import types

# Same prompt used in production (gemini_analyzer.py)
PROMPT = """Telecamera fissa DENTRO un garage. L'immagine può essere a colori (giorno)
o IR bianco/nero (notte). La porta sezionale è al centro. Determina: APERTA o CHIUSA.

CLOSED — i pannelli coprono il varco:
  - Pannelli orizzontali chiari formano una superficie VERTICALE continua
  - Giunture/linee scure tra i pannelli (è normale, NON è un'apertura)
  - Binario curvo visibile in alto ai lati
  - Il varco è completamente sigillato, non si vede nulla oltre la porta

OPEN — il varco è libero, la porta è retratta:
  - I pannelli NON sono una superficie verticale: sono ripiegati/retratti
    lungo il soffitto (visibili come struttura orizzontale in alto)
  - Il varco mostra un'apertura ampia: si vede rampa, esterno, pergola, cielo
  - Anche parzialmente aperta (pannelli a metà, varco parziale in basso) = OPEN

COME DISTINGUERE (la regola fondamentale):
  Guarda il VARCO della porta (la zona rettangolare al centro dell'immagine).
  → Se è coperto da pannelli piatti verticali con linee orizzontali → CLOSED
  → Se mostra un'apertura (totale o parziale) verso l'esterno → OPEN
  Nel dubbio → CLOSED.

Rispondi SOLO in JSON: {"status": "open"|"closed", "confidence": 0.0-1.0, "reasoning": "..."}
Il campo reasoning deve essere in italiano, massimo 10 parole.
Se l'immagine è troppo ambigua, imposta confidence sotto 0.5."""

MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

# Paid tier Standard pricing (per 1M tokens)
PRICING = {
    "gemini-2.0-flash-lite": {"input": 0.075, "output": 0.30},
    "gemini-2.0-flash":      {"input": 0.10,  "output": 0.40},
    "gemini-2.5-flash-lite": {"input": 0.10,  "output": 0.40},
    "gemini-2.5-flash":      {"input": 0.30,  "output": 2.50},
}


def analyze(client, model, image_bytes):
    """Send image to model, return parsed result dict."""
    start = time.time()
    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                PROMPT,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )
        elapsed = time.time() - start

        data = json.loads(response.text)
        usage = getattr(response, "usage_metadata", None)
        input_tok = getattr(usage, "prompt_token_count", 0) or 0
        output_tok = getattr(usage, "candidates_token_count", 0) or 0

        prices = PRICING.get(model, {"input": 0, "output": 0})
        cost = (input_tok * prices["input"] + output_tok * prices["output"]) / 1_000_000

        return {
            "status": data.get("status", "?"),
            "confidence": float(data.get("confidence", 0)),
            "reasoning": data.get("reasoning", ""),
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "cost": cost,
            "latency": elapsed,
            "error": None,
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "status": "ERROR",
            "confidence": 0,
            "reasoning": str(e)[:60],
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0,
            "latency": elapsed,
            "error": str(e),
        }


def main():
    api_key = os.environ.get("GM_GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GM_GEMINI_API_KEY environment variable")
        sys.exit(1)

    # Collect image files from args or default paths
    if len(sys.argv) > 1:
        image_paths = sys.argv[1:]
    else:
        print("Usage: python scripts/compare_models.py image1.jpg image2.jpg ...")
        sys.exit(1)

    # Validate files
    images = {}
    for path in image_paths:
        if not os.path.isfile(path):
            print(f"WARNING: File not found: {path}, skipping")
            continue
        name = os.path.basename(path)
        with open(path, "rb") as f:
            images[name] = f.read()
        print(f"Loaded: {name} ({len(images[name]):,} bytes)")

    if not images:
        print("No valid images found.")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    print(f"\nModels: {', '.join(MODELS)}")
    print(f"Images: {len(images)}")
    print(f"Total API calls: {len(images) * len(MODELS)}")
    print()

    # Run all combinations
    results = {}  # (image_name, model) -> result
    for img_name, img_bytes in images.items():
        for model in MODELS:
            key = (img_name, model)
            print(f"  Testing {model} on {img_name}...", end=" ", flush=True)
            result = analyze(client, model, img_bytes)
            results[key] = result
            status_icon = "O" if result["status"] == "open" else ("C" if result["status"] == "closed" else "?")
            print(
                f"{status_icon} conf={result['confidence']:.0%} "
                f"in={result['input_tokens']} out={result['output_tokens']} "
                f"${result['cost']:.6f} {result['latency']:.1f}s"
            )
            time.sleep(0.5)  # rate limit courtesy
        print()

    # Summary table per image
    print("=" * 120)
    print("RESULTS SUMMARY")
    print("=" * 120)

    for img_name in images:
        print(f"\n{'─' * 120}")
        print(f"  {img_name}")
        print(f"{'─' * 120}")
        print(
            f"  {'Model':<28} {'Status':<8} {'Conf':>6} "
            f"{'In tok':>8} {'Out tok':>8} {'Cost':>10} {'Latency':>8}  Reasoning"
        )
        print(f"  {'─' * 110}")
        for model in MODELS:
            r = results[(img_name, model)]
            print(
                f"  {model:<28} {r['status']:<8} {r['confidence']:>5.0%} "
                f"{r['input_tokens']:>8} {r['output_tokens']:>8} "
                f"${r['cost']:>8.6f} {r['latency']:>7.1f}s  {r['reasoning']}"
            )

    # Cost comparison (projected monthly)
    print(f"\n{'=' * 120}")
    print("MONTHLY COST PROJECTION (3,000 calls/month, based on avg tokens per model)")
    print(f"{'=' * 120}")
    print(f"  {'Model':<28} {'Avg In':>8} {'Avg Out':>8} {'$/call':>10} {'$/month':>10}")
    print(f"  {'─' * 70}")
    for model in MODELS:
        model_results = [results[(img, model)] for img in images if not results[(img, model)]["error"]]
        if not model_results:
            continue
        avg_in = sum(r["input_tokens"] for r in model_results) / len(model_results)
        avg_out = sum(r["output_tokens"] for r in model_results) / len(model_results)
        avg_cost = sum(r["cost"] for r in model_results) / len(model_results)
        monthly = avg_cost * 3000
        print(
            f"  {model:<28} {avg_in:>8.0f} {avg_out:>8.0f} "
            f"${avg_cost:>9.6f} ${monthly:>9.2f}"
        )

    print()


if __name__ == "__main__":
    main()
