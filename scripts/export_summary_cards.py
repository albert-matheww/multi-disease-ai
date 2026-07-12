"""dump a short markdown summary card per disease after training.

reads model metadata and emits the compact cards used on the
streamlit about page.
"""
import json
from pathlib import Path

def main():
    for name in ["diabetes", "ckd", "liver", "heart"]:
        meta = json.loads((Path("models") / f"{name}_meta.json").read_text())
        card = f"""### {meta['name']}

- model: {meta['model']}
- best f1: {meta['best_f1']}
- features: {len(meta['features'])}
"""
        out = Path("reports/cards")
        out.mkdir(exist_ok=True)
        (out / f"{name}.md").write_text(card)
        print("wrote", name)

if __name__ == "__main__":
    main()
