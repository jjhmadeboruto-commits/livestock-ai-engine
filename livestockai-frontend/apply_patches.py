"""Apply Base44 frontend patches to the downloaded live bundle."""
from pathlib import Path

JS = Path(__file__).parent / "assets" / "index-DN-wla8G.js"
OUT = Path(__file__).parent / "assets" / "index-DN-wla8G.patched.js"

OLD_CJ = (
    'async function cJ(t){const e=new FormData;e.append("image",t);let r;try{'
    "r=await fetch(`${LN}/api/estimate-weight`,{method:\"POST\",body:e})"
    "}catch{throw new Error(`Cannot reach backend at ${LN}. Is your Flask server running?`)}"
    'let n;try{n=await r.json()}catch{throw new Error("Backend returned a non-JSON response.")}'
    'if(!r.ok||n.status!=="success")throw new Error((n==null?void 0:n.message)||`Backend error: HTTP ${r.status}`);return n}'
)

NEW_CJ = (
    'async function cJ(t,e){const a=new FormData;a.append("image",t);'
    'const o={Cattle:"dairy_cow",Pig:"pig",Poultry:"poultry",Goat:"goat",Sheep:"sheep",Donkey:"donkey"}[e]||"dairy_cow";'
    "let r;try{r=await fetch(`${LN}/api/estimate-weight?animal_type=${o}`,"
    '{method:"POST",body:a,signal:AbortSignal.timeout(9e4)})'
    "}catch{throw new Error(`Cannot reach backend at ${LN}. Is your Flask server running?`)}"
    'let n;try{n=await r.json()}catch{throw new Error("Backend returned a non-JSON response.")}'
    'if(!r.ok||n.status!=="success")throw new Error((n==null?void 0:n.message)||`Backend error: HTTP ${r.status}`);return n}'
)

REPLACEMENTS = [
    (
        "MediaPipe pose detection estimates body length and height instantly.",
        "Computer vision estimates body length and height from your photo.",
    ),
    ("Analysing morphology\u2026", "Analyzing image\u2026"),
    ("MediaPipe pose detection running", "Analyzing image"),
    ("Detecting skeletal keypoints", "Detecting animal in frame"),
    (OLD_CJ, NEW_CJ),
    ("const j=await cJ(x);", "const j=await cJ(x,_);"),
]


def main() -> None:
    js = JS.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        if old not in js:
            raise SystemExit(f"Patch target not found:\n  {old[:80]}...")
        js = js.replace(old, new, 1)
        print("patched:", old[:50])

    OUT.write_text(js, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
