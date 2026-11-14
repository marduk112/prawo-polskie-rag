"""
Sprawdza w ELI API Sejmu, czy ustawy w bazie RAG (data/raw/*_meta.json)
są nadal aktualne -- czyli czy od czasu ostatniego uruchomienia
download_acts.py:
1. akt nie został uchylony (status / inForce / "Akty uchylające"),
2. nie pojawił się nowszy tekst ujednolicony (zmiana nazwy pliku typu
   "U" w metadanych aktu -- oznacza nowelizację, której jeszcze nie
   mamy w data/processed/).

To zautomatyzowana wersja ręcznej kontroli, która przy kroku 5
(PROGRESS.md) wykryła, że "ustawa o promocji zatrudnienia i
instytucjach rynku pracy" (DU/2004/1001) została uchylona i zastąpiona
nową ustawą (DU/2025/620) -- bez tej kontroli dodalibyśmy do bazy
nieaktualny akt.

Rozróżnia PROBLEMY (kod wyjścia 1) od INFORMACJI (nie psują wyniku):
- problem: akt uchylony/nieobowiązujący (chyba że w acts_config.json ma
  `"historical": true` -- akt celowo trzymany jako poprzednik, np. do `--as-of`),
  albo nowszy tekst ujednolicony;
- informacja: zmieniła się tylko data `changeDate` w metadanych (bywa tak
  przy poprawkach metadanych i nowelizacjach jeszcze bez tekstu jednolitego --
  sprawdź ręcznie, jeśli zależy Ci na najświeższym stanie).

Nie modyfikuje żadnych danych -- tylko raportuje. Jeśli coś jest nieaktualne,
napraw ręcznie (zaktualizuj ACTS w download_acts.py, jeśli trzeba, i
uruchom ponownie download_acts.py + build_rag_index.py).

Użycie:
    python scripts/check_acts_freshness.py
    python scripts/check_acts_freshness.py --domain podatkowe
"""

import argparse
import sys
from pathlib import Path

import requests

from download_acts import API_BASE, RAW_DIR, load_acts_config, pick_text_file

OK_STATUSES = {"obowiązujący", "akt posiada tekst jednolity"}


def load_local_meta(short: str) -> dict | None:
    path = RAW_DIR / f"{short}_meta.json"
    if not path.exists():
        return None
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def check_act(act: dict) -> tuple[list[str], list[str]]:
    """Zwraca (problemy, informacje); pusta lista problemów = wszystko w porządku."""
    problems: list[str] = []
    infos: list[str] = []
    short = act["short"]
    historical = bool(act.get("historical"))

    local = load_local_meta(short)
    if local is None:
        problems.append("brak lokalnych metadanych -- uruchom najpierw download_acts.py")
        return problems, infos

    url = f"{API_BASE}/acts/{act['publisher']}/{act['year']}/{act['position']}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    remote = resp.json()

    status = remote.get("status", "")
    in_force = remote.get("inForce", "")
    if not historical and (status not in OK_STATUSES or in_force != "IN_FORCE"):
        problems.append(
            f"akt nie wygląda na obowiązujący (status='{status}', inForce='{in_force}')"
        )

    repealing = remote.get("references", {}).get("Akty uchylające")
    if repealing and not historical:
        successors = ", ".join(r["id"] for r in repealing)
        problems.append(f"akt UCHYLONY -- następca(y): {successors}")

    try:
        _, remote_file = pick_text_file(remote)
        _, local_file = pick_text_file(local)
        if remote_file != local_file:
            problems.append(
                f"pojawił się nowszy tekst ujednolicony ({local_file} -> {remote_file}) "
                "-- prawdopodobna nowelizacja, dane w RAG mogą być nieaktualne"
            )
    except RuntimeError as e:
        problems.append(f"nie można porównać tekstów: {e}")

    remote_change = remote.get("changeDate", "")
    local_change = local.get("changeDate", "")
    if remote_change and local_change and remote_change > local_change:
        infos.append(
            f"metadane aktu zmieniły się od ostatniego pobrania "
            f"({local_change} -> {remote_change}) -- bez nowego tekstu ujednoliconego; "
            "sprawdź ręcznie, jeśli zależy Ci na najświeższym stanie"
        )

    return problems, infos


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Ogranicz do jednej dziedziny (pole 'domains' w data/acts_config.json). "
        "Domyślnie: wszystkie akty z konfiguracji.",
    )
    args = parser.parse_args()

    any_problems = False
    for act in load_acts_config(domain=args.domain):
        print(f"[{act['short']}] sprawdzam...")
        try:
            problems, infos = check_act(act)
        except requests.RequestException as e:
            problems, infos = [f"błąd zapytania do ELI API: {e}"], []

        if not problems and not infos:
            print("  OK -- aktualne")
        for p in problems:
            any_problems = True
            print(f"  UWAGA: {p}")
        for i in infos:
            print(f"  INFO: {i}")
        print()

    if any_problems:
        print("Znaleziono akty wymagające uwagi -- patrz UWAGA wyżej.")
        sys.exit(1)
    else:
        print("Brak problemów (wpisy INFO nie psują wyniku).")


if __name__ == "__main__":
    main()
