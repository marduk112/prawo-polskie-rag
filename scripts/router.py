"""Router dziedzin oparty o Basal (https://github.com/rkinas/basal) -- wybiera
dziedzinę (indeks RAG) na podstawie samego pytania, zamiast ręcznego --domain.

Basal działa jako osobny serwer HTTP (`basal-serve`, patrz README/PROGRESS.md);
ten moduł jest tylko lekkim klientem (sama biblioteka standardowa, bez torch/mlx),
żeby dało się go importować i testować bez ładowania modeli.

Pomiar (PROGRESS.md / pamięć projektu, 2026-10-25, opisy dziedzin = nazwy ustaw z
acts_config.json, basal-2.0-4.5B-MLX-8bit): ścisła trafność 77,6% (bateria 147
pytań) i 96,5% (zbiór kontrolny 113 pytań), właściwa dziedzina w top-2 w 93-99%.
To za mało na twardy router, dlatego przy pewności poniżej progu zwracamy też
drugą dziedzinę (wyszukiwanie w obu indeksach i scalenie wyników).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
ACTS_CONFIG_PATH = ROOT / "data" / "acts_config.json"

DEFAULT_URL = "http://localhost:8000/v1/systemone"
# Powyżej tej pewności szukamy w jednej dziedzinie, poniżej -- w dwóch. Krok 100: na baterii 129
# (retrieval przez router, Bielik 11B) próg 0,85 dał 116 OK, 0,9 -- 113, 0,8 -- 113 (przy 32
# pytaniach z drugą dziedziną zamiast 52), a bez drugiej dziedziny 104. Różnica 3 przypadków jest
# dobrana na tej samej baterii, więc to słaby dowód -- do potwierdzenia na pytaniach spoza zbioru.
DEFAULT_SINGLE_THRESHOLD = 0.85

QUESTION_INSTRUCTIONS = "Do której dziedziny prawa należy to pytanie użytkownika?"


class RouterError(RuntimeError):
    """Serwer Basala niedostępny albo zwrócił nieoczekiwaną odpowiedź."""


@dataclass(frozen=True)
class RouteDecision:
    primary: str
    secondary: str | None  # None = wystarczy jedna dziedzina
    confidence: float
    probabilities: dict[str, float]

    @property
    def domains(self) -> list[str]:
        return [self.primary] + ([self.secondary] if self.secondary else [])


def build_criteria(acts_config: list[dict], domains: list[str] | None = None) -> dict[str, str]:
    """Opis dziedziny = nazwy ustaw w niej (z acts_config.json). Ręcznie napisane,
    tematyczne opisy NIE poprawiły wyniku w pomiarze (a wydłużały prompt), więc
    zostajemy przy nazwach ustaw. `domains`: ogranicza do dziedzin z zbudowanym
    indeksem RAG (router nie może wskazać dziedziny, której nie ma jak przeszukać)."""
    acts_by_domain: dict[str, set[str]] = {}
    for act in acts_config:
        for domain in act["domains"]:
            acts_by_domain.setdefault(domain, set()).add(act["short"])
    if domains is not None:
        acts_by_domain = {d: v for d, v in acts_by_domain.items() if d in set(domains)}
    return {
        domain: "Prawo polskie: " + ", ".join(s.replace("_", " ") for s in sorted(shorts))
        for domain, shorts in acts_by_domain.items()
    }


def decide(
    probabilities: dict[str, float], confidence: float, single_threshold: float = DEFAULT_SINGLE_THRESHOLD
) -> RouteDecision:
    """Czysta logika wyboru (bez HTTP): pewność >= próg -> jedna dziedzina, w przeciwnym
    razie dwie o najwyższym prawdopodobieństwie. Remisy rozstrzyga nazwa (deterministycznie)."""
    if not probabilities:
        raise RouterError("Basal nie zwrócił prawdopodobieństw dziedzin")
    ranked = sorted(probabilities, key=lambda d: (-probabilities[d], d))
    secondary = None if confidence >= single_threshold or len(ranked) < 2 else ranked[1]
    return RouteDecision(ranked[0], secondary, confidence, dict(probabilities))


def _post_json(url: str, body: dict, timeout: float) -> dict:
    request = Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except (URLError, OSError) as exc:
        raise RouterError(
            f"Nie można połączyć się z serwerem Basala ({url}): {exc}. Uruchom go np.: "
            "basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --port 8000"
        ) from exc


def route(
    question: str,
    criteria: dict[str, str],
    url: str = DEFAULT_URL,
    single_threshold: float = DEFAULT_SINGLE_THRESHOLD,
    timeout: float = 120.0,
    post: Callable[[str, dict, float], dict] = _post_json,
) -> RouteDecision:
    """Zadaje serwerowi Basala jedno pytanie `choice` o dziedzinę. `post` do podmiany w testach."""
    body = {
        "state": question,
        "questions": {"dziedzina": {"type": "choice", "instructions": QUESTION_INSTRUCTIONS, "criteria": criteria}},
    }
    response = post(url, body, timeout)
    try:
        answer = response["answers"]["dziedzina"]
        return decide(answer["probabilities"], float(answer["confidence"]), single_threshold)
    except (KeyError, TypeError, ValueError) as exc:
        raise RouterError(f"Nieoczekiwana odpowiedź Basala: {str(response)[:200]}") from exc


def load_acts_config(path: Path = ACTS_CONFIG_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def available_domains(processed_dir: Path | None = None) -> list[str]:
    """Dziedziny z zbudowanym indeksem RAG (rag_index_<domena>.npy + _meta.json)."""
    processed_dir = processed_dir or ROOT / "data" / "processed"
    return sorted(
        path.name.removeprefix("rag_index_").removesuffix("_meta.json")
        for path in processed_dir.glob("rag_index_*_meta.json")
        if (processed_dir / path.name.replace("_meta.json", ".npy")).exists()
    )


def auto_domain(question: str, url: str = DEFAULT_URL, **route_kwargs) -> RouteDecision:
    """`--domain auto`: wariant minimalny -- bierzemy tylko `primary`; `secondary`
    (niska pewność) jest zwracane w decyzji, ale wywołujący je na razie ignoruje."""
    criteria = build_criteria(load_acts_config(), available_domains())
    return route(question, criteria, url=url, **route_kwargs)


def effective_domains(decision: dict, secondary_below: float = DEFAULT_SINGLE_THRESHOLD) -> list[str]:
    """Dziedziny do przeszukania z zapisanej decyzji (wiersz z route_battery): `primary`, plus
    `secondary` gdy router jest mniej pewny niż `secondary_below` (> 1 = zawsze)."""
    secondary = decision.get("secondary")
    if secondary and decision["confidence"] < secondary_below:
        return [decision["primary"], secondary]
    return [decision["primary"]]


def route_battery(cases: list[dict], out_path: Path, url: str = DEFAULT_URL, **route_kwargs) -> dict[str, dict]:
    """Routuje pytania z przypadków referencyjnych (pola `id`, `question`) i zapisuje
    {id: {primary, secondary, confidence}} -- wejście dla
    check_expand_query_regressions.py --routing-file."""
    criteria = build_criteria(load_acts_config(), available_domains())
    routing = {}
    for case in cases:
        d = route(case["question"], criteria, url=url, **route_kwargs)
        routing[case["id"]] = {
            "primary": d.primary,
            "secondary": d.secondary,
            "confidence": d.confidence,
            "probabilities": d.probabilities,  # pozwala liczyć inne progi offline
        }
    out_path.write_text(json.dumps(routing, ensure_ascii=False, indent=1), encoding="utf-8")
    return routing


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Router dziedzin (Basal): pytanie albo bateria referencyjna")
    parser.add_argument("question", nargs="*", help="Pytanie (bez --battery)")
    parser.add_argument("--battery", type=Path, default=None, metavar="WYNIK.json",
                        help="Zrouting wszystkich przypadków z data/expand_query_reference_cases.json do pliku")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--cases", type=Path, default=ROOT / "data" / "expand_query_reference_cases.json",
                        help="Plik z przypadkami (domyślnie bateria 129; np. data/routing_holdout_cases.json)")
    parser.add_argument("--single-threshold", type=float, default=DEFAULT_SINGLE_THRESHOLD,
                        help="Próg pewności; > 1 = zawsze zapisuj drugą dziedzinę (do liczenia progów offline)")
    args = parser.parse_args()
    if args.battery:
        cases = json.loads(args.cases.read_text(encoding="utf-8"))
        routing = route_battery(cases, args.battery, url=args.url, single_threshold=args.single_threshold)
        print(f"Zapisano {len(routing)} decyzji do {args.battery}")
    elif args.question:
        print(auto_domain(" ".join(args.question), url=args.url))
    else:
        parser.error("podaj pytanie albo --battery")


if __name__ == "__main__":
    main()
