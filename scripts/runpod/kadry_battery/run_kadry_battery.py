"""Bateria end-to-end repo kadrowego z Qwenem: prawdziwy chat_lmstudio.answer() (RAG repo kadrowego + prompt + parametry Qwena)
przeciw vLLM. Pytania: 26 z etykietami artykułu (data/retrieval_battery.json) + 76 z Kroków 106/115. Historia pusta (jedno pytanie = jedna tura).
Wynik: results.json + podsumowanie. Wątki tylko dla HTTP; wyszukiwanie pod blokadą."""
import json, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import chat_lmstudio as cl
from file_index import SessionFileIndex
from rag_search import RagIndex

URL, MODEL, WORKERS = "http://127.0.0.1:8000/v1", "qwen3.8-27b-fp8", 8
labeled = json.loads((ROOT / "data/retrieval_battery.json").read_text(encoding="utf-8"))
extra = json.loads((ROOT / "data/battery_questions_76.json").read_text(encoding="utf-8"))
items = [{"set": "etykiety26", "id": b["id"], "question": b["question"], "act": b["act_short"], "article": b["article"]} for b in labeled]
seen = {i["question"] for i in items}
items += [{"set": "pytania76", "id": f"q{n}", "question": q, "act": None, "article": None} for n, q in enumerate(extra) if q not in seen]

rag = RagIndex()
files = SessionFileIndex(model=rag.model)
max_tokens, reasoning = cl.resolve_generation_defaults(MODEL, None, None)
print(f"[INFO] model={MODEL} max_tokens={max_tokens} reasoning_effort={reasoning} pytań={len(items)}", flush=True)

lock, tl = threading.Lock(), threading.local()
orig_search, orig_post = cl.search_with_history, cl.requests.post

def locked_search(*a, **k):
    with lock:
        res = orig_search(*a, **k)
    tl.retrieved = [(h.get("act_short"), str(h.get("article"))) for h in res]
    return res

def capture_post(url, *a, **k):
    resp = orig_post(url, *a, **k)
    try:
        j = resp.json(); tl.finish = j["choices"][0].get("finish_reason"); tl.tokens = (j.get("usage") or {}).get("completion_tokens")
    except Exception:
        pass
    return resp

cl.search_with_history, cl.requests.post = locked_search, capture_post
ART = re.compile(r"[Aa]rt\.\s*(\d+[a-ząćęłńóśźż]{0,3}[⁰¹²³⁴⁵⁶⁷⁸⁹]*)")

def run(it):
    tl.retrieved, tl.finish, tl.tokens = [], None, None
    t0 = time.time(); text, err = "", None
    try:
        text = cl.answer(URL, MODEL, rag, files, [], it["question"], 8, max_tokens=max_tokens, reasoning_effort=reasoning)
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
    cited = sorted({m.group(1) for m in ART.finditer(text)})
    ret_numbers = {a for _, a in tl.retrieved}
    r = {**it, "answer": text, "error": err, "seconds": round(time.time() - t0, 1), "finish": tl.finish, "tokens": tl.tokens,
         "words": len(text.split()), "retrieved": tl.retrieved, "cited": cited,
         "cited_not_in_context": [c for c in cited if c not in ret_numbers]}
    if it["article"]:
        r["expected_in_context"] = any(act == it["act"] and a == it["article"] for act, a in tl.retrieved)
        r["expected_cited"] = it["article"] in cited
    print(f"[{it['set']}] {it['id']}: {r['seconds']}s fin={r['finish']} słów={r['words']} err={bool(err)}", flush=True)
    return r

with ThreadPoolExecutor(WORKERS) as pool:
    results = list(pool.map(run, items))
(ROOT / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

n = len(results); errs = [r for r in results if r["error"]]; trunc = [r for r in results if r["finish"] == "length"]
lab = [r for r in results if r["article"]]
print("\n=== PODSUMOWANIE ===")
print(f"pytań: {n}, błędy/puste: {len(errs)}, ucięte (finish=length): {len(trunc)}")
print(f"średnio: {sum(r['seconds'] for r in results)/n:.1f} s/odp (8 równolegle), {sum(r['words'] for r in results)/n:.0f} słów")
print(f"z etykietą ({len(lab)}): artykuł w kontekście {sum(r['expected_in_context'] for r in lab)}/{len(lab)}, cytowany w odpowiedzi {sum(r['expected_cited'] for r in lab)}/{len(lab)}")
print(f"odpowiedzi cytujące numer artykułu spoza kontekstu (heurystyka): {sum(bool(r['cited_not_in_context']) for r in results)}/{n}")
