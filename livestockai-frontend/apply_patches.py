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
    (OLD_CJ, NEW_CJ),
    ("const j=await cJ(x);", "const j=await cJ(x,_);"),
    (
        'const j=await cJ(x,_);n({...j,filename:x.name,animalType:_}),p({id:Date.now(),timestamp:new Date().toISOString().replace("T"," ").slice(0,16),filename:x.name,animalType:_,weight:j.estimated_weight_kg,confidence:j.confidence_interval,bodyLength:j.body_length_cm,bodyHeight:j.body_height_cm,pendingSync:!navigator.onLine})',
        'const j=await cJ(x,_);const o={dairy_cow:"Cattle",beef_cattle:"Cattle",young_cattle:"Cattle",pig:"Pig",poultry:"Poultry",goat:"Goat",sheep:"Sheep",donkey:"Donkey"}[j.detected_species_key]||_;n({...j,filename:x.name,animalType:o}),p({id:Date.now(),timestamp:new Date().toISOString().replace("T"," ").slice(0,16),filename:x.name,animalType:o,weight:j.estimated_weight_kg,confidence:j.confidence_interval,bodyLength:j.body_length_cm,bodyHeight:j.body_height_cm,pendingSync:!navigator.onLine})'
    ),
    (
        'visible_health_concerns&&I.jsxs("p",{className:"text-xs pt-1",children:[I.jsx("span",{style:{color:"#aaa"},children:"Health: "}),I.jsx("span",{style:{color:t.visible_health_concerns.toLowerCase().includes("none")?"#16a34a":"#dc2626",fontWeight:600},children:t.visible_health_concerns})]}),t.ai_attribution',
        'visible_health_concerns&&I.jsxs("p",{className:"text-xs pt-1",children:[I.jsx("span",{style:{color:"#aaa"},children:"Health: "}),I.jsx("span",{style:{color:t.visible_health_concerns.toLowerCase().includes("none")?"#16a34a":"#dc2626",fontWeight:600},children:t.visible_health_concerns})]}),t.ai_self_review&&I.jsxs("div",{className:"p-3 rounded-lg text-xs",style:{background:"rgba(245,158,11,0.06)",border:"1px solid rgba(245,158,11,0.2)"},children:[I.jsxs("div",{className:"flex items-center gap-1.5 mb-1.5 font-semibold",style:{color:"#b45309"},children:[I.jsx("span",{children:"🤔"}),I.jsx("span",{children:"AI Self-Review"})]}),I.jsx("p",{className:"italic leading-relaxed",style:{color:"#78350f"},children:t.ai_self_review})]}),t.ai_attribution'
    ),
]


def main() -> None:
    js = JS.read_text(encoding="utf-8")
    for old, new in REPLACEMENTS:
        if old not in js:
            print(f"SKIP (already applied): {old[:60]}...")
            continue
        js = js.replace(old, new, 1)
        print("patched:", old[:60])

    OUT.write_text(js, encoding="utf-8")
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
