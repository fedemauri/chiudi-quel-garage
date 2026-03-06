"""Test the new Gemini prompt against sample images."""

import glob
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from dotenv import load_dotenv
load_dotenv()

from garage_monitor.gemini_analyzer import GeminiAnalyzer, PROMPT

EXPECTED = {
    "IMG_6517": "closed",  # IR notte, porta chiusa (nuova posizione)
    "IMG_6518": "open",    # IR notte, porta aperta (nuova posizione)
    "IMG_6519": "closed",  # IR notte, porta chiusa (vecchia posizione)
    "IMG_6520": "open",    # diurna, porta aperta (vecchia posizione)
    "Screenshot 2026-03-06 (00.34.53)": "closed",  # falso positivo originale
}

def main():
    api_key = os.environ["GM_GEMINI_API_KEY"]
    model = os.environ.get("GM_GEMINI_MODEL", "gemini-2.5-flash")
    analyzer = GeminiAnalyzer(api_key, model)

    print(f"Modello: {model}")
    print(f"Prompt:\n{PROMPT}\n")
    print("=" * 60)

    images = sorted(glob.glob(os.path.expanduser("~/Downloads/IMG_651*.JPG")) +
                    glob.glob(os.path.expanduser("~/Downloads/IMG_652*.JPG")) +
                    glob.glob(os.path.expanduser("~/Downloads/Screenshot 2026-03-06*")))
    if not images:
        print("Nessuna immagine trovata in ~/Downloads/IMG_651*.JPG")
        return

    results = []
    for path in images:
        name = os.path.splitext(os.path.basename(path))[0]
        expected = EXPECTED.get(name, "?")

        with open(path, "rb") as f:
            image_bytes = f.read()

        result = analyzer.analyze(image_bytes)
        ok = result.status.value == expected
        symbol = "OK" if ok else "FAIL"

        results.append((name, expected, result.status.value, result.confidence, result.reasoning, ok))
        print(f"[{symbol}] {name}: atteso={expected}, ottenuto={result.status.value}, "
              f"confidence={result.confidence:.2f}, reasoning=\"{result.reasoning}\"")

    print("=" * 60)
    passed = sum(1 for r in results if r[5])
    print(f"Risultato: {passed}/{len(results)} corretti")


if __name__ == "__main__":
    main()
