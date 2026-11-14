"""Destylacja sekwencyjna Qwen -> Bielik (PROGRESS.md, Krok 72).

Cztery etapy, każdy wznawialny (już policzone rekordy są pomijane), wyniki w
data/distill/<dziedzina>/:

  questions  Qwen (serwer zgodny z OpenAI API, np. vLLM na RunPodzie) generuje
             potoczne pytania do każdego bieżącego artykułu dziedziny.
             -> questions.jsonl
  contexts   LOKALNIE, Bielik MLX bez adaptera: expand_query + RAG dokładnie
             jak w chat.py (ten sam prompt systemowy, ten sam build_user_message,
             top_k=8) -- uczeń w treningu widzi to samo, co przy użyciu.
             Pytania prawie identyczne z pytaniami ewaluacyjnymi (baterie,
             EVAL_QUESTIONS niżej) są odrzucane (kosinus embeddingów >= 0.85).
             -> prompts.jsonl
  answers    Qwen odpowiada na DOKŁADNIE te same wiadomości (system + kontekst +
             pytanie), równolegle. -> answers.jsonl
  dataset    filtr (kompletna odpowiedź, cytowane artykuły w kontekście, limit
             długości w tokenach Bielika) + podział train/valid w formacie
             mlx_lm (messages: system, user z kontekstem, assistant) ->
             data/finetune/<dziedzina>_distill/{train,valid}.jsonl

Różnica względem dotychczasowych danych fine-tune (data/finetune/<dziedzina>/):
tamte mają samo pytanie -> odpowiedź, BEZ kontekstu RAG, podczas gdy adapter
zawsze działa z kontekstem -- uczy się odpowiadać z pamięci zamiast z
dostarczonych przepisów.

Klucz API serwera: zmienna środowiskowa LLM_SERVER_API_KEY (vLLM --api-key; ta sama
co w chat_lmstudio.py).

Użycie (etap contexts ładuje Bielika -- jeden proces MLX naraz):
    python scripts/distill_qwen.py questions --url https://<pod>-8000.proxy.runpod.net/v1 --domain konsumenckie
    python scripts/distill_qwen.py contexts --domain konsumenckie
    python scripts/distill_qwen.py answers --url ... --domain konsumenckie
    python scripts/distill_qwen.py dataset --domain konsumenckie
"""

import argparse
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from chat_lmstudio import _headers, post_chat  # noqa: E402

DISTILL_DIR = ROOT / "data" / "distill"
FINETUNE_DIR = ROOT / "data" / "finetune"
CITE_RE = re.compile(r"art\.\s*(\d+[a-ząćęłńóśźż]{0,3}[⁰¹²³⁴⁵⁶⁷⁸⁹]{0,2})", re.IGNORECASE)
NEAR_DUPLICATE_COSINE = 0.85
# Qwen czasem przełącza język w środku słowa ("co dokładnie承诺owano",
# "nadmiernыми" -- Krok 72); taki przykład uczyłby ucznia śmieci.
FOREIGN_SCRIPT_RE = re.compile(r"[\u0400-\u04ff\u0600-\u06ff\u3000-\u9fff\uac00-\ud7af]")
MAX_ANSWER_TOKENS = 480  # chat.py generuje max 500 tokenów -- dłuższa odpowiedź nauczyciela uczyłaby ucinania
# Cały przykład (system + kontekst + pytanie + odpowiedź) w tokenach Bielika --
# max_seq_length treningu (mlx_lm i train_lora_cuda.py --max-length). Dłuższy
# zostałby obcięty od końca, czyli razem z odpowiedzią, na której liczy się
# strata (Krok 76: 101/482 przykładów budowlanych miało 11-17 tys. tokenów).
# Domyślnie limit lokalnego treningu MLX; na CUDA (96 GB) --max-example-tokens 16384.
MAX_EXAMPLE_TOKENS = 11264

# Pytania ewaluacyjne spoza baterii -- trzymane z dala od zbioru treningowego.
EVAL_QUESTIONS = {
    "konsumenckie": [
        "Pralka po naprawie dalej cieknie. Czy mogę teraz odstąpić od umowy i dostać zwrot pieniędzy?",
        "Odesłałem kurtkę kupioną przez internet tydzień temu, a sklep dalej nie oddał pieniędzy. Ile ma na to czasu?",
        "Sklep nie odpowiedział na moją reklamację od trzech tygodni. Co to oznacza?",
        "Zamówiłem stół robiony na wymiar według moich wymiarów. Czy mogę go zwrócić w ciągu 14 dni?",
        "Kupiłem telefon przez internet, rozpakowałem i używałem przez tydzień, a teraz chcę go oddać. Czy sklep może obniżyć zwrot za zużycie?",
        "Sprzedawca odesłał mi naprawiony laptop, ale wada wróciła po miesiącu. Co mogę zrobić?",
        "Akwizytor przyszedł do domu i sprzedał mi odkurzacz za 5 tysięcy. Czy mogę się wycofać po trzech tygodniach?",
        "Sklep internetowy każe mi zapłacić za odesłanie zwracanego towaru. Czy to zgodne z prawem?",
        "Kupiłem bilet na koncert przez internet i nie mogę pójść. Czy mogę odstąpić od umowy w ciągu 14 dni?",
        "Zamówiłem kanapę, ale przyszła w innym kolorze niż na zdjęciu. Czego mogę żądać?",
    ],
}

# Dopisywane TYLKO do zapytania do nauczyciela, nie do zapisanego przykładu
# treningowego -- uczeń uczy się zwięzłego stylu na dokładnie tym wejściu, które
# zobaczy w chat.py (generuje max 500 tokenów). Bez tego Qwen pisze 1,5-3 tys.
# znaków z nagłówkami i tabelami (pierwsza próba, Krok 72).
TEACHER_STYLE_NOTE = (
    "\n\n[Forma odpowiedzi: zwięźle, najwyżej ok. 180 słów, zwykłym tekstem w 1-3 akapitach, "
    "bez nagłówków, tabel, list wypunktowanych i pogrubień. Na początku jasna odpowiedź, potem "
    "podstawa prawna z numerem artykułu (ustępu, punktu) z dostarczonych fragmentów, a jeśli "
    "odpowiedź zależy od okoliczności niepodanych w pytaniu albo fragmenty sprawy nie rozstrzygają "
    "-- powiedz to wprost. Cytuj wyłącznie przepisy, które są w dostarczonych fragmentach -- "
    "nie dopisuj artykułów z pamięci (Krok 77: 12% odpowiedzi w `cywilne` odpadało przez cytat "
    "spoza kontekstu). Bez formułek o tym, że to nie jest porada prawna.]"
)

QUESTION_GEN_SYSTEM = (
    "Jesteś autorem realistycznych pytań, jakie zwykli ludzie (nie prawnicy) zadają asystentowi "
    "prawnemu. Piszesz po polsku, potocznie, z konkretną sytuacją życiową."
)
QUESTION_GEN_USER = """Poniżej przepis z ustawy "{act_title}".

{text}

Napisz {n} różnych pytań, które mógłby zadać zwykły człowiek w sytuacji, której dotyczy ten przepis.
Wymagania:
- potoczny język, konkretna sytuacja (co kupił, gdzie, kiedy), bez numerów artykułów i bez cytowania ustawy;
- różne sytuacje i różne sformułowania, nie parafrazy jednego pytania;
- co najmniej jedno pytanie takie, w którym odpowiedź zależy od okoliczności niepodanych w pytaniu albo przepis rozstrzyga sprawę tylko częściowo;
- jeśli przepis jest czysto techniczny (np. o wejściu w życie, zmianach innych ustaw, definicjach bez praktycznego znaczenia), zwróć pustą listę.

Zwróć WYŁĄCZNIE tablicę JSON z pytaniami jako napisami, bez komentarza."""


def domain_dir(domain: str) -> Path:
    d = DISTILL_DIR / domain
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    try:
        return [json.loads(line) for line in lines]
    except json.JSONDecodeError:
        # Ucięta ostatnia linia po przerwanym procesie (Krok 76) -- wznowienie ją policzy od nowa.
        recs = [json.loads(line) for line in lines[:-1]]
        print(f"[UWAGA] {path.name}: pominięta uszkodzona ostatnia linia", flush=True)
        return recs


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class Teacher:
    def __init__(self, url: str, model: str | None):
        self.url = url.rstrip("/")
        if model is None:
            resp = requests.get(f"{self.url}/models", headers=_headers(), timeout=30)
            resp.raise_for_status()
            model = resp.json()["data"][0]["id"]
        self.model = model

    def chat(self, messages: list[dict], max_tokens: int, temperature: float, thinking: bool) -> dict:
        kwargs = {"reasoning_effort": "low"} if thinking else {"enable_thinking": False}
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "chat_template_kwargs": kwargs,
        }
        res = post_chat(self.url, payload, timeout=1800)  # strumieniowo -- patrz post_chat (błąd 524 proxy)
        return {**res, "content": res["content"].strip()}


def current_articles(domain: str) -> list[dict]:
    articles = json.loads((ROOT / "data" / "processed" / "all_articles.json").read_text(encoding="utf-8"))
    return [a for a in articles if domain in a.get("domains", [])]


def parse_question_list(text: str) -> list[str]:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        items = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    return [q.strip() for q in items if isinstance(q, str) and len(q.strip()) > 15]


def cmd_questions(args) -> None:
    out = domain_dir(args.domain) / "questions.jsonl"
    done = {r["source_article"] for r in read_jsonl(out)}
    teacher = Teacher(args.url, args.model)
    todo = [
        a for a in current_articles(args.domain)
        if a["id"] not in done and len(a["text"]) >= 150 and not re.search(r"\((uchylony|pominięte)\)", a["text"][:80])
    ]
    print(f"Model: {teacher.model}; artykułów do przetworzenia: {len(todo)}")

    def work(a: dict) -> tuple[dict, list[str]]:
        user = QUESTION_GEN_USER.format(act_title=a["act_title"], text=a["text"][:6000], n=args.per_article)
        r = teacher.chat(
            [{"role": "system", "content": QUESTION_GEN_SYSTEM}, {"role": "user", "content": user}],
            max_tokens=1500, temperature=0.7, thinking=False,
        )
        return a, parse_question_list(r["content"])

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for fut in as_completed([pool.submit(work, a) for a in todo]):
            a, questions = fut.result()
            for q in questions:
                append_jsonl(out, {"source_article": a["id"], "act_short": a["act_short"], "article": a["article"], "question": q})
            if not questions:  # znacznik "przetworzony", żeby wznowienie go pominęło
                append_jsonl(out, {"source_article": a["id"], "act_short": a["act_short"], "article": a["article"], "question": None})
            print(f"  art. {a['article']}: {len(questions)} pytań", flush=True)


def eval_questions(domain: str) -> list[str]:
    qs = list(EVAL_QUESTIONS.get(domain, []))
    for name in ("expand_query_reference_cases.json", "qwen_answer_reference_cases.json"):
        for case in json.loads((ROOT / "data" / name).read_text(encoding="utf-8")):
            if case.get("domain") == domain or domain in json.dumps(case, ensure_ascii=False):
                q = case.get("question") or (case.get("turns") or [None])[-1]
                if isinstance(q, str):
                    qs.append(q)
    # Pytania z porównania adapter vs baza (Krok 73) -- mają notatkę 'expect' do
    # ręcznej oceny, więc też trzymane z dala od treningu (Krok 76).
    compare = json.loads((ROOT / "data" / "adapter_vs_base_questions.json").read_text(encoding="utf-8"))
    qs.extend(item["q"] for item in compare.get(domain, []))
    # Pytania z większego porównania adapter vs baza (Krok 88, 76 pytań kadr) -- Krok 121: 13 z nich
    # występowało dosłownie w odpowiedziach treningowych, więc też muszą być trzymane z dala od treningu.
    compare_file = DISTILL_DIR / domain / "compare_questions.json"
    if compare_file.exists():
        qs.extend(json.loads(compare_file.read_text(encoding="utf-8")))
    # Pytania z odłożonych artykułów (cmd_holdout, Krok 77).
    holdout = holdout_path(domain)
    if holdout.exists():
        qs.extend(json.loads(holdout.read_text(encoding="utf-8"))["questions"])
    return qs


def holdout_path(domain: str) -> Path:
    return DISTILL_DIR / domain / "eval_holdout.json"


def cmd_holdout(args) -> None:
    """Odkłada N losowych artykułów do ewaluacji: po jednym pytaniu z każdego
    trafia do eval_questions, a WSZYSTKIE pytania z tych artykułów wypadają z
    treningu (cmd_contexts) -- ewaluacja na przepisach, których uczeń nie
    widział w żadnym przykładzie."""
    out = holdout_path(args.domain)
    if out.exists():
        print(f"{out} już istnieje -- bez zmian")
        return
    by_article: dict[str, list[str]] = {}
    for r in read_jsonl(domain_dir(args.domain) / "questions.jsonl"):
        if r["question"]:  # artykuły techniczne mają rekord bez pytania (Krok 78)
            by_article.setdefault(r["source_article"], []).append(r["question"])
    chosen = sorted(random.Random(0).sample(sorted(by_article), min(args.articles, len(by_article))))
    data = {"articles": chosen, "questions": [by_article[a][0] for a in chosen]}
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Odłożone artykuły: {len(chosen)} -> {out}")


def cmd_contexts(args) -> None:
    import numpy as np

    from prompt import build_query_expansion_messages, build_system_prompt, build_user_message, search_with_history
    from rag_search import QUERY_PREFIX, RagIndex, release_accelerator_cache

    out = domain_dir(args.domain) / "prompts.jsonl"
    done = {r["question"] for r in read_jsonl(out)}
    questions = [r for r in read_jsonl(domain_dir(args.domain) / "questions.jsonl") if r["question"]]
    holdout = holdout_path(args.domain)
    if holdout.exists():
        held = set(json.loads(holdout.read_text(encoding="utf-8"))["articles"])
        questions = [r for r in questions if r["source_article"] not in held]
        print(f"Bez pytań z {len(held)} odłożonych artykułów: {len(questions)}")
    rag = RagIndex(domain=args.domain)

    def embed(texts: list[str]) -> np.ndarray:
        return rag.model.encode([QUERY_PREFIX + t for t in texts], normalize_embeddings=True)

    evals = eval_questions(args.domain)
    eval_emb = embed(evals)
    q_emb = embed([r["question"] for r in questions])
    sims = q_emb @ eval_emb.T
    kept, dropped = [], 0
    for r, row in zip(questions, sims):
        if row.max() >= NEAR_DUPLICATE_COSINE:
            dropped += 1
            print(f"  [odrzucone, blisko pytania ewaluacyjnego {row.max():.2f}] {r['question'][:90]}")
        else:
            kept.append(r)
    print(f"Pytań: {len(questions)}, odrzuconych jako bliskie ewaluacji: {dropped}, pytań ewaluacyjnych: {len(evals)}")
    if args.sample and len(kept) > args.sample:
        # Losowy podzbiór (stałe ziarno -- wznowienie wybiera ten sam), gdy
        # pytań jest więcej, niż zmieści budżet treningu (cywilne, Krok 76).
        kept = sorted(random.Random(0).sample(kept, args.sample), key=questions.index)
        print(f"Losowy podzbiór: {len(kept)} pytań")
    todo = [r for r in kept if r["question"] not in done]

    system = build_system_prompt(args.domain)
    act_titles = rag.act_titles()

    def save(i: int, r: dict, expanded: str) -> None:
        q = r["question"]
        extra = [expanded] if expanded and expanded.strip().lower() != q.strip().lower() else None
        hits = search_with_history(rag, [], q, 8, extra_queries=extra)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": build_user_message(q, hits, [])}]
        append_jsonl(out, {
            **r,
            "expanded": expanded,
            "context_articles": [[h["act_short"], h["article"]] for h in hits],
            "messages": messages,
        })
        print(f"  [{i}/{len(todo)}] {q[:70]}", flush=True)

    if args.expand_url:
        # Bielik na serwerze zgodnym z OpenAI API (vLLM na podzie, Krok 76):
        # ten sam prompt co chat.expand_query, zachłannie (temperature 0, jak
        # domyślny sampler mlx_lm), max 80 tokenów. Ekspansje równolegle w
        # wątkach, wyszukiwanie po kolei w głównym wątku (jeden model
        # embeddingów/reranker na GPU).
        bielik = Teacher(args.expand_url, None)

        def expand(r: dict) -> tuple[dict, str]:
            messages = build_query_expansion_messages(r["question"], act_titles=act_titles, domain=args.domain)
            payload = {"model": bielik.model, "messages": messages, "max_tokens": 80, "temperature": 0.0}
            for attempt in range(3):
                try:
                    return r, post_chat(bielik.url, payload, timeout=300)["content"].strip()
                except requests.RequestException as e:
                    print(f"  [ekspansja, próba {attempt + 1}/3] {type(e).__name__}: {r['question'][:50]}", flush=True)
            return r, ""  # bez ekspansji -- samo pytanie, jak chat.py przy pustej ekspansji

        print(f"Bielik (ekspansja): {bielik.model}; do policzenia: {len(todo)}")
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            for i, fut in enumerate(as_completed([pool.submit(expand, r) for r in todo]), 1):
                save(i, *fut.result())
        return

    from mlx_lm import load

    from chat import MODEL_PATH, expand_query

    model, tokenizer = load(str(MODEL_PATH))
    for i, r in enumerate(todo, 1):
        expanded = expand_query(model, tokenizer, r["question"], lora_layers=None, act_titles=act_titles, domain=args.domain)
        save(i, r, expanded)
        release_accelerator_cache()


def cmd_answers(args) -> None:
    out = domain_dir(args.domain) / "answers.jsonl"
    done = {r["question"] for r in read_jsonl(out)}
    prompts = read_jsonl(domain_dir(args.domain) / "prompts.jsonl")
    if args.sample and len(prompts) > args.sample:
        # Losowy podzbiór, nie N pierwszych -- prompts.jsonl idzie po kolei po
        # artykułach, więc --limit pokryłby tylko początek ustawy. Odpowiedzi
        # spoza podzbioru policzone wcześniej zostają w answers.jsonl (Krok 76:
        # na L40S ~7 odpowiedzi/min, prompty po ~11 tys. tokenów).
        prompts = random.Random(0).sample(prompts, args.sample)
    todo = [r for r in prompts if r["question"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    teacher = Teacher(args.url, args.model)
    print(f"Model: {teacher.model}; do policzenia: {len(todo)}")

    def work(r: dict) -> dict | None:
        # Zerwany strumień (proxy RunPoda tnie połączenie po ~100 s bez bajtu,
        # np. gdy zapytanie czeka w kolejce vLLM) -- ponów zamiast tracić partię.
        messages = r["messages"][:-1] + [
            {**r["messages"][-1], "content": r["messages"][-1]["content"] + TEACHER_STYLE_NOTE}
        ]
        for attempt in range(3):
            try:
                res = teacher.chat(messages, max_tokens=args.max_tokens, temperature=0.2, thinking=True)
                return {**r, "teacher": res}
            except requests.RequestException as e:
                print(f"  [próba {attempt + 1}/3 nieudana] {type(e).__name__}: {r['question'][:50]}", flush=True)
        return None

    failed = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        for n, fut in enumerate(as_completed([pool.submit(work, r) for r in todo]), 1):
            rec = fut.result()
            if rec is None:
                failed += 1
                continue
            append_jsonl(out, rec)
            t = rec["teacher"]
            print(f"  [{n}/{len(todo)}] {t['finish_reason']} out={t['usage'].get('completion_tokens')} {rec['question'][:60]}", flush=True)
    print(f"Gotowe; nieudanych po 3 próbach: {failed} (wznowienie je dobierze)", flush=True)


def cited_outside_context(answer: str, context_articles: list[list[str]], context_text: str = "") -> list[str]:
    """Artykuły cytowane w odpowiedzi, których nie ma w kontekście -- ani jako
    fragment, ani jako odesłanie w treści fragmentu (np. art. 27 sam odsyła do
    art. 33, 34 i 35; powtórzenie tego odesłania to nie wiedza spoza kontekstu)."""
    allowed = {a for _, a in context_articles} | {m.group(1) for m in CITE_RE.finditer(context_text)}
    return sorted({m.group(1) for m in CITE_RE.finditer(answer)} - allowed)


def near_eval_duplicates(domain: str, questions: list[str]) -> set[str]:
    """Pytania (z `questions`) identyczne lub bliskie (kosinus >= NEAR_DUPLICATE_COSINE) którymś pytaniu
    ewaluacyjnemu -- jak filtr w cmd_contexts, ale też dla już policzonych odpowiedzi (Krok 121)."""
    from rag_search import QUERY_PREFIX, RagIndex

    evals = eval_questions(domain)
    exact = {q for q in questions if q in set(evals)}
    rest = [q for q in questions if q not in exact]
    if not rest or not evals:
        return exact
    model = RagIndex(domain=domain).model
    sims = model.encode([QUERY_PREFIX + q for q in rest], normalize_embeddings=True) @ model.encode(
        [QUERY_PREFIX + q for q in evals], normalize_embeddings=True
    ).T
    return exact | {q for q, row in zip(rest, sims) if row.max() >= NEAR_DUPLICATE_COSINE}


def cmd_dataset(args) -> None:
    from transformers import AutoTokenizer

    if args.tokenizer:  # np. repo HF na podzie bez MLX (Krok 76)
        tokenizer_path = args.tokenizer
    else:
        from chat import MODEL_PATH

        tokenizer_path = str(MODEL_PATH)
    # Sam tokenizer, bez ładowania wag -- nie zajmuje GPU/MLX (można odpalić
    # równolegle z innym procesem MLX).
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    recs = read_jsonl(domain_dir(args.domain) / "answers.jsonl")
    kept, reasons = [], {}
    near_eval = near_eval_duplicates(args.domain, [r["question"] for r in recs])
    for r in recs:
        t = r["teacher"]
        answer = re.sub(r"<think>.*?</think>", "", t["content"], flags=re.DOTALL).strip()
        reason = None
        if r["question"] in near_eval:
            reason = "bliskie pytaniu ewaluacyjnemu"
        elif t["finish_reason"] != "stop" or not answer:
            reason = "niekompletna"
        elif FOREIGN_SCRIPT_RE.search(answer):
            reason = "obce pismo"
        elif cited_outside_context(answer, r["context_articles"], r["messages"][-1]["content"]):
            reason = "cytat spoza kontekstu"
        elif len(tokenizer.encode(answer)) > MAX_ANSWER_TOKENS:
            reason = "za długa"
        elif len(tokenizer(tokenizer.apply_chat_template(
            r["messages"] + [{"role": "assistant", "content": answer}], tokenize=False
        ))["input_ids"]) > args.max_example_tokens:
            reason = "za długi przykład"
        if reason:
            reasons[reason] = reasons.get(reason, 0) + 1
            continue
        kept.append({"messages": r["messages"] + [{"role": "assistant", "content": answer}], "source_article": r["source_article"]})
    print(f"Odpowiedzi: {len(recs)}, zachowane: {len(kept)}, odrzucone: {reasons}")

    # Podział po ARTYKUŁACH źródłowych, nie po pytaniach -- pytania do tego
    # samego artykułu są do siebie podobne, więc losowy podział zawyżałby
    # walidację.
    sources = sorted({k["source_article"] for k in kept})
    random.Random(0).shuffle(sources)
    valid_sources = set(sources[: max(1, round(len(sources) * args.valid_frac))])
    target = FINETUNE_DIR / f"{args.domain}_distill"
    target.mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "valid": 0}
    with (target / "train.jsonl").open("w", encoding="utf-8") as ftrain, (target / "valid.jsonl").open("w", encoding="utf-8") as fvalid:
        for k in kept:
            split = "valid" if k["source_article"] in valid_sources else "train"
            (fvalid if split == "valid" else ftrain).write(json.dumps({"messages": k["messages"]}, ensure_ascii=False) + "\n")
            counts[split] += 1
    print(f"Zapisano {target}: {counts}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("questions", "holdout", "contexts", "answers", "dataset"):
        p = sub.add_parser(name)
        p.add_argument("--domain", required=True)
        if name in ("questions", "answers"):
            p.add_argument("--url", required=True, help="Bazowy URL serwera zgodnego z OpenAI API (z /v1)")
            p.add_argument("--model", default=None, help="ID modelu (domyślnie: pierwszy z /models)")
            p.add_argument("--concurrency", type=int, default=16, help="vLLM na A40: >16 zapełnia KV cache i tworzy kolejkę -> zerwania przez proxy")
        if name == "questions":
            p.add_argument("--per-article", type=int, default=4)
        if name == "answers":
            p.add_argument("--max-tokens", type=int, default=4000)
            p.add_argument("--limit", type=int, default=0, help="Tylko N pierwszych (próba)")
            p.add_argument("--sample", type=int, default=0, help="Losowy podzbiór N promptów (0 = wszystkie)")
        if name == "holdout":
            p.add_argument("--articles", type=int, default=12)
        if name == "contexts":
            p.add_argument("--expand-url", default=None, help="Bielik na serwerze OpenAI API (vLLM) zamiast lokalnego MLX")
            p.add_argument("--concurrency", type=int, default=32)
            p.add_argument("--sample", type=int, default=0, help="Losowy podzbiór N pytań (0 = wszystkie)")
        if name == "dataset":
            p.add_argument("--valid-frac", type=float, default=0.1)
            p.add_argument("--max-example-tokens", type=int, default=MAX_EXAMPLE_TOKENS, help="= --max-length treningu")
            p.add_argument("--tokenizer", default=None, help="Ścieżka/repo tokenizera Bielika (domyślnie lokalny model MLX z chat.py)")
    args = parser.parse_args()
    {"questions": cmd_questions, "holdout": cmd_holdout, "contexts": cmd_contexts, "answers": cmd_answers, "dataset": cmd_dataset}[args.cmd](args)


if __name__ == "__main__":
    main()
