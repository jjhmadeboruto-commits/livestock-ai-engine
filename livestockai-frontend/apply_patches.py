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
    (
        'const j=await cJ(x,_);n({...j,filename:x.name,animalType:_}),p({id:Date.now(),timestamp:new Date().toISOString().replace("T"," ").slice(0,16),filename:x.name,animalType:_,weight:j.estimated_weight_kg,confidence:j.confidence_interval,bodyLength:j.body_length_cm,bodyHeight:j.body_height_cm,pendingSync:!navigator.onLine})',
        'const j=await cJ(x,_);const o={dairy_cow:"Cattle",beef_cattle:"Cattle",young_cattle:"Cattle",pig:"Pig",poultry:"Poultry",goat:"Goat",sheep:"Sheep",donkey:"Donkey"}[j.detected_species_key]||_;n({...j,filename:x.name,animalType:o}),p({id:Date.now(),timestamp:new Date().toISOString().replace("T"," ").slice(0,16),filename:x.name,animalType:o,weight:j.estimated_weight_kg,confidence:j.confidence_interval,bodyLength:j.body_length_cm,bodyHeight:j.body_height_cm,pendingSync:!navigator.onLine})'
    ),
    (
        'I.jsxs("div",{className:"grid grid-cols-2 gap-3",children:[I.jsxs("div",{className:"card-clean p-4",children:[I.jsxs("div",{className:"flex items-center gap-1.5 text-xs mb-1.5",style:{color:"#aaa"},children:[I.jsx(s8,{size:11,color:"#16a34a"})," Body Length"]}),I.jsxs("div",{className:"text-2xl font-bold font-serif",style:{color:"#141414",fontFamily:"\'Playfair Display\', serif"},children:[n," ",I.jsx("span",{className:"text-xs font-normal",style:{color:"#bbb"},children:"cm"})]})]}),I.jsxs("div",{className:"card-clean p-4",children:[I.jsxs("div",{className:"flex items-center gap-1.5 text-xs mb-1.5",style:{color:"#aaa"},children:[I.jsx(XT,{size:11,color:"#16a34a"})," Body Height"]}),I.jsxs("div",{className:"text-2xl font-bold font-serif",style:{color:"#141414",fontFamily:"\'Playfair Display\', serif"},children:[i," ",I.jsx("span",{className:"text-xs font-normal",style:{color:"#bbb"},children:"cm"})]})]})]})',
        'I.jsxs("div",{className:"grid grid-cols-2 gap-3",children:[I.jsxs("div",{className:"card-clean p-4",children:[I.jsxs("div",{className:"flex items-center gap-1.5 text-xs mb-1.5",style:{color:"#aaa"},children:[I.jsx(s8,{size:11,color:"#16a34a"})," Body Length"]}),I.jsxs("div",{className:"text-2xl font-bold font-serif",style:{color:"#141414",fontFamily:"\'Playfair Display\', serif"},children:[n," ",I.jsx("span",{className:"text-xs font-normal",style:{color:"#bbb"},children:"cm"})]})]}),I.jsxs("div",{className:"card-clean p-4",children:[I.jsxs("div",{className:"flex items-center gap-1.5 text-xs mb-1.5",style:{color:"#aaa"},children:[I.jsx(XT,{size:11,color:"#16a34a"})," Body Height"]}),I.jsxs("div",{className:"text-2xl font-bold font-serif",style:{color:"#141414",fontFamily:"\'Playfair Display\', serif"},children:[i," ",I.jsx("span",{className:"text-xs font-normal",style:{color:"#bbb"},children:"cm"})]})]})]}),I.jsxs("div",{className:"card-clean p-5 space-y-4",style:{background:"#fdfdfb",border:"1px solid rgba(22,163,74,0.15)"},children:[I.jsxs("div",{className:"flex items-center gap-2 pb-2",style:{borderBottom:"1px solid rgba(0,0,0,0.05)"},children:[I.jsx("span",{className:"text-lg",children:"✨"}),I.jsx("h3",{className:"font-serif font-bold text-sm",style:{color:"#141414",fontFamily:"\'Playfair Display\', serif"},children:"Gemini Visual Intelligence"})]}),t.gemini_explanation&&I.jsx("p",{className:"text-xs italic leading-relaxed",style:{color:"#444"},children:t.gemini_explanation}),I.jsxs("div",{className:"grid grid-cols-2 gap-2.5 text-xs pt-1",children:[I.jsxs("div",{children:[I.jsx("span",{style:{color:"#aaa"},children:"Breed: "}),I.jsx("span",{className:"font-semibold",style:{color:"#141414"},children:t.breed||"Unknown"})]}),I.jsxs("div",{children:[I.jsx("span",{style:{color:"#aaa"},children:"Sex: "}),I.jsx("span",{className:"font-semibold",style:{color:"#141414"},children:t.sex||"Unknown"})]}),I.jsxs("div",{children:[I.jsx("span",{style:{color:"#aaa"},children:"Age: "}),I.jsx("span",{className:"font-semibold",style:{color:"#141414"},children:t.estimated_age_months?t.estimated_age_months+" mo":"Unknown"})]}),I.jsxs("div",{children:[I.jsx("span",{style:{color:"#aaa"},children:"BCS: "}),I.jsx("span",{className:"font-semibold",style:{color:"#141414"},children:t.body_condition?t.body_condition+(t.body_condition_score?" (Score "+t.body_condition_score+")":""):"Not assessed"})]})]}),t.visible_health_concerns&&I.jsxs("p",{className:"text-xs pt-1",children:[I.jsx("span",{style:{color:"#aaa"},children:"Health: "}),I.jsx("span",{style:{color:t.visible_health_concerns.toLowerCase().includes("none")?"#16a34a":"#dc2626",fontWeight:600},children:t.visible_health_concerns})]}),t.ai_attribution&&I.jsxs("div",{className:"text-[10px] pt-2 flex flex-col gap-0.5",style:{color:"#aaa",borderTop:"1px solid rgba(0,0,0,0.04)"},children:[I.jsxs("p",{children:[I.jsx("span",{children:"🔍 Detection: "}),I.jsx("span",{style:{color:"#666"},children:t.ai_attribution.detection_model||"YOLOv8"})]}),I.jsxs("p",{children:[I.jsx("span",{children:"🧬 Classification: "}),I.jsx("span",{style:{color:"#666"},children:t.ai_attribution.classification_model||"CLIP"})]}),I.jsxs("p",{children:[I.jsx("span",{children:"🧠 AI Engine: "}),I.jsx("span",{style:{color:"#16a34a",fontWeight:600},children:t.ai_attribution.enrichment_model||"Google Gemini 2.5 Flash"})]}),I.jsxs("p",{children:[I.jsx("span",{children:"⚖️ Weight Blend: "}),I.jsx("span",{style:{color:"#666"},children:t.ai_attribution.weight_blend||"Formula + Visual intelligence"})]})]})]})'
    ),
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
