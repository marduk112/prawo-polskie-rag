"""
Eksport CZYSTEGO drzewa do publikacji (Krok 125) -- bez pushowania czegokolwiek.

Z prywatnego repo (pełne archiwum, ze wszystkimi danymi i historią) buduje w
osobnym katalogu kopię nadającą się do upublicznienia:

1. Bierze WYŁĄCZNIE pliki śledzone w gicie (`git archive HEAD`): nic nieśledzonego
   (adaptery, modele, data/processed, lokalne ustawienia) nie ma jak się tam znaleźć.
2. Pomija dane i logi, które nie mają iść na zewnątrz (domyślnie data/distill,
   data/finetune -- odpowiedzi Qwena i logi podów; patrz PROGRESS.md, Krok 125).
3. Czyści ścieżki lokalne w plikach tekstowych (/Users/<ty>/..., scratchpad Claude Code).
4. Skanuje wynik pod kątem sekretów, adresów e-mail i resztek ścieżek; kończy się
   kodem 1, jeśli znajdzie sekret albo e-mail (ścieżki i inne uwagi są tylko raportowane).
5. Opcjonalnie: świeża historia git (jeden commit, wskazany autor), odpalenie testów.

Skrypt NIE zmienia repo źródłowego i NIE robi `git push`. Wynik to katalog do
przeglądu przez człowieka -- skrypt niczego nie publikuje.

Użycie:
    python scripts/export_public.py ../rag-prawo-public
    python scripts/export_public.py ../rag-prawo-public --no-distill-scripts
    python scripts/export_public.py ../rag-prawo-public --run-tests
    python scripts/export_public.py ../rag-prawo-public --init-git \\
        --author "Imię <12345+login@users.noreply.github.com>"
    python scripts/export_public.py --dry-run          # tylko lista i rozmiary, bez zapisu

Opcje:
    --exclude WZORZEC     dodatkowy pathspec do pominięcia (można powtarzać), np. data/raw
    --keep-data           NIE pomijaj data/distill i data/finetune (świadoma decyzja)
    --no-distill-scripts  pomiń też skrypty destylacji/ewaluacji adapterów (DISTILL_SCRIPTS)
"""

import argparse
import io
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Domyślnie pomijane (pathspec względem korzenia repo).
DEFAULT_EXCLUDES = [
    "data/distill",  # odpowiedzi Qwena + logi podów RunPod
    "data/finetune",  # ręczne zbiory iter25 i zbiory *_distill z kontekstem RAG
]

# Skrypty służące wyłącznie przepisowi destylacji/adapterów (zamkniętemu w Krokach 121-122)
# -- pomijane tylko z --no-distill-scripts.
DISTILL_SCRIPTS = [
    "scripts/runpod",
    "scripts/distill_qwen.py",
    "scripts/eval_distill.py",
    "scripts/eval_distill_vllm.py",
    "scripts/judge_distill.py",
    "scripts/train_lora_cuda.py",
    "scripts/convert_lora.py",
    "scripts/check_lora_converter.py",
    "scripts/compare_adapter_vs_base.py",
    "scripts/test_hedge_probes_mlx.sh",
    "scripts/test_hedge_probes_cuda.sh",
    "tests/test_convert_lora.py",
    "tests/test_chat_adapter_guard.py",
    "data/adapter_vs_base_comparison.json",
    "data/adapter_vs_base_questions.json",
]

TEXT_SUFFIXES = {".py", ".md", ".sh", ".json", ".jsonl", ".yml", ".yaml", ".txt", ".toml", ".ini", ".log", ".out", ""}

# Zamiany ścieżek lokalnych (kolejność ma znaczenie: od najdłuższych/najbardziej szczegółowych).
PATH_SUBSTITUTIONS = [
    (re.compile(r"/Users/[A-Za-z0-9._-]+/Projekty/RAG-LoRA - prawo polskie"), "<repo>"),
    (re.compile(r"/private/tmp/claude-\d+/[^\s\"')`]*"), "<scratchpad>"),
    (re.compile(r"/Users/[A-Za-z0-9._-]+/\.claude/[^\s\"')`]*"), "<lokalny-plik-planu>"),
    (re.compile(r"/Users/[A-Za-z0-9._-]+/"), "~/"),
]

# Skan: SEKRET lub E-MAIL => błąd (kod 1); reszta => uwaga.
SECRET_PATTERNS = {
    "token HF": re.compile(r"hf_[A-Za-z0-9]{25,}"),
    "klucz RunPod": re.compile(r"rpa_[A-Za-z0-9]{15,}"),
    "klucz prywatny": re.compile(r"BEGIN (RSA |OPENSSH |EC )?PRIVATE KEY"),
    "klucz API (sk-)": re.compile(r"sk-[A-Za-z0-9]{20,}"),
    "przypisanie klucza": re.compile(r"""(api[_-]?key|secret|password|token)\s*[=:]\s*['"][A-Za-z0-9_\-]{16,}['"]""", re.I),
}
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@(?!users\.noreply\.github\.com)[A-Za-z0-9.-]+\.[a-z]{2,}")
# Adresy, które wolno zostawić (przykłady, kontakt w licencjach itp.)
EMAIL_ALLOW = re.compile(r"(@example\.|noreply@|@anthropic\.com|git@github\.com)")
NOTE_PATTERNS = {
    "ścieżka lokalna": re.compile(r"/Users/|/private/tmp/claude|\bLexar\b|/Volumes/"),
    "adres IP (poza loopbackiem)": re.compile(r"\b(?!127\.0\.0\.1\b|0\.0\.0\.0\b)(?:\d{1,3}\.){3}\d{1,3}\b"),
}


def run(cmd, **kw):
    return subprocess.run(cmd, cwd=ROOT, check=True, capture_output=True, **kw)


def tracked_size(pathspec: str) -> tuple[int, int]:
    """(liczba plików, bajty) śledzonych w HEAD pod pathspec -- do raportu."""
    out = run(["git", "ls-tree", "-r", "-l", "HEAD", "--", pathspec]).stdout.decode()
    n = total = 0
    for line in out.splitlines():
        parts = line.split(None, 4)
        if len(parts) >= 4 and parts[3].isdigit():
            n += 1
            total += int(parts[3])
    return n, total


def human(b: int) -> str:
    return f"{b / 1e6:.1f} MB"


def export_tree(out: Path, excludes: list[str]) -> None:
    cmd = ["git", "archive", "--format=tar", "HEAD", "--", "."] + [f":(exclude){e}" for e in excludes]
    data = run(cmd).stdout
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        tar.extractall(out, filter="data")


def scrub_paths(out: Path) -> dict[str, int]:
    changed: dict[str, int] = {}
    for p in out.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new, n_total = text, 0
        for rx, repl in PATH_SUBSTITUTIONS:
            new, n = rx.subn(repl, new)
            n_total += n
        if n_total:
            p.write_text(new, encoding="utf-8")
            changed[str(p.relative_to(out))] = n_total
    return changed


def scan(out: Path) -> tuple[list[str], list[str]]:
    """Zwraca (błędy, uwagi). Błędy: sekrety i adresy e-mail."""
    errors: list[str] = []
    notes: dict[str, list[str]] = {k: [] for k in NOTE_PATTERNS}
    for p in out.rglob("*"):
        if not p.is_file() or p.stat().st_size > 20_000_000:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        rel = str(p.relative_to(out))
        if rel == "scripts/export_public.py":
            continue  # zawiera wzorce skanu (siebie samego nie skanujemy)
        for name, rx in SECRET_PATTERNS.items():
            for m in rx.finditer(text):
                errors.append(f"SEKRET ({name}): {rel}: {m.group(0)[:12]}...")
        for m in EMAIL_PATTERN.finditer(text):
            if not EMAIL_ALLOW.search(m.group(0)):
                errors.append(f"E-MAIL: {rel}: {m.group(0)}")
        for name, rx in NOTE_PATTERNS.items():
            n = len(rx.findall(text))
            if n:
                notes[name].append(f"{rel} ({n})")
    uwagi = [f"{k}: {len(v)} plików -- " + ", ".join(v[:8]) + (" ..." if len(v) > 8 else "") for k, v in notes.items() if v]
    return errors, uwagi


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", type=Path, help="Katalog wyjściowy (musi nie istnieć albo być pusty)")
    ap.add_argument("--exclude", action="append", default=[], help="Dodatkowy pathspec do pominięcia")
    ap.add_argument("--keep-data", action="store_true", help="Nie pomijaj data/distill ani data/finetune")
    ap.add_argument("--no-distill-scripts", action="store_true", help="Pomiń skrypty destylacji/adapterów")
    ap.add_argument("--run-tests", action="store_true", help="Po eksporcie: pytest -m 'not slow' w wyniku")
    ap.add_argument("--init-git", action="store_true", help="Świeża historia: git init + jeden commit")
    ap.add_argument("--author", type=str, default=None, help='Autor commitu: "Imię <adres-noreply>" (z --init-git)')
    ap.add_argument("--dry-run", action="store_true", help="Tylko raport rozmiarów, bez zapisu")
    args = ap.parse_args()

    if args.init_git and not args.author:
        ap.error("--init-git wymaga --author (adres noreply GitHuba, nie prywatny)")
    if args.author and not re.fullmatch(r".+ <[^<>@\s]+@[^<>@\s]+>", args.author):
        ap.error('--author musi mieć postać "Imię <adres>"')

    excludes = ([] if args.keep_data else list(DEFAULT_EXCLUDES)) + args.exclude
    if args.no_distill_scripts:
        excludes += DISTILL_SCRIPTS

    dirty = run(["git", "status", "--porcelain", "--untracked-files=no"]).stdout.decode().strip()
    if dirty:
        print("UWAGA: repo źródłowe ma niezacommitowane zmiany -- eksport bierze HEAD, nie katalog roboczy:")
        print("  " + dirty.replace("\n", "\n  "))

    print("Pomijane (rozmiar śledzony w HEAD):")
    for e in excludes:
        n, b = tracked_size(e)
        print(f"  {e:<45} {n:>4} plików  {human(b)}")
    n_all, b_all = tracked_size(".")
    print(f"Razem śledzone: {n_all} plików, {human(b_all)}")
    if args.dry_run:
        return 0

    if args.out is None:
        ap.error("podaj katalog wyjściowy (albo --dry-run)")
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        ap.error("katalog wyjściowy nie może leżeć wewnątrz repo źródłowego")
    if out.exists() and any(out.iterdir()):
        ap.error(f"{out} nie jest pusty -- nic nie nadpisuję")

    export_tree(out, excludes)
    n_out = sum(1 for p in out.rglob("*") if p.is_file())
    b_out = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"\nWyeksportowano: {n_out} plików, {human(b_out)} -> {out}")

    scrubbed = scrub_paths(out)
    print(f"Oczyszczono ścieżki lokalne w {len(scrubbed)} plikach: " + ", ".join(f"{k} ({v})" for k, v in list(scrubbed.items())[:10]))

    errors, uwagi = scan(out)
    print("\nUwagi do przeglądu (nie blokują):")
    for u in uwagi or ["  brak"]:
        print(f"  - {u}")

    if args.run_tests:
        print("\nTesty w wyeksportowanym drzewie (pytest -m 'not slow'):")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not slow", "-p", "no:cacheprovider"], cwd=out)
        if r.returncode != 0:
            errors.append(f"testy w wyeksportowanym drzewie nie przechodzą (kod {r.returncode})")
        shutil.rmtree(out / ".pytest_cache", ignore_errors=True)
        for pc in out.rglob("__pycache__"):
            shutil.rmtree(pc, ignore_errors=True)

    if args.init_git:
        env_author = args.author
        name, mail = re.fullmatch(r"(.+) <(.+)>", env_author).groups()
        for c in (["git", "init", "-q", "-b", "main"], ["git", "add", "-A"]):
            subprocess.run(c, cwd=out, check=True)
        subprocess.run(
            ["git", "-c", f"user.name={name}", "-c", f"user.email={mail}", "commit", "-q", "-m", "Pierwsze wydanie publiczne"],
            cwd=out, check=True,
        )
        print(f"\nŚwieża historia: 1 commit, autor {env_author}. Zdalnego repo NIE ustawiono i nic nie wypchnięto.")

    if errors:
        print("\nBŁĘDY (do poprawienia przed publikacją):")
        for e in errors:
            print(f"  ! {e}")
        return 1
    print("\nSkan: brak sekretów i prywatnych adresów e-mail. Przejrzyj uwagi powyżej i LICENSE/README przed publikacją.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
