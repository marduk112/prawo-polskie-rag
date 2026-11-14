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

Tryb `--update <klon-publicznego-repo>` (kolejne aktualizacje): buduje i skanuje
drzewo dokładnie tak samo, w katalogu tymczasowym, a potem synchronizuje je
z istniejącym klonem publicznego repo i robi w nim ZWYKŁY commit (historia rośnie,
nie powstaje nowa). Do publicznego repo trafia wyłącznie to, co przeszło filtr
(wykluczenia, czyszczenie ścieżek, skan) -- dane i logi spoza listy nigdy.
Źródłem prawdy jest repo prywatne: zmiany zrobione bezpośrednio w klonie
publicznym zostaną nadpisane (lista zmian jest widoczna w `--dry-run`).

Skrypt NIE zmienia repo źródłowego i NIE robi `git push`. Wynik to katalog (albo
commit w klonie) do przeglądu przez człowieka -- skrypt niczego nie publikuje.

Użycie:
    python scripts/export_public.py ../rag-prawo-public
    python scripts/export_public.py ../rag-prawo-public --no-distill-scripts
    python scripts/export_public.py ../rag-prawo-public --run-tests
    python scripts/export_public.py ../rag-prawo-public --init-git \\
        --author "Imię <12345+login@users.noreply.github.com>"
    python scripts/export_public.py --dry-run          # tylko lista i rozmiary, bez zapisu
    python scripts/export_public.py --update ../rag-prawo-public --dry-run
    python scripts/export_public.py --update ../rag-prawo-public \\
        --author "Imię <12345+login@users.noreply.github.com>" --message "Opis zmian"

Opcje:
    --exclude WZORZEC     dodatkowy pathspec do pominięcia (można powtarzać), np. data/raw
    --keep-data           NIE pomijaj data/distill i data/finetune (świadoma decyzja)
    --no-distill-scripts  pomiń też skrypty destylacji/ewaluacji adapterów (DISTILL_SCRIPTS)
    --update KLON         zaktualizuj istniejący klon publicznego repo zwykłym commitem
    --message TEKST       komunikat commitu dla --update (domyślnie: numer wersji archiwum)
"""

import argparse
import io
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
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


def build_clean_tree(out: Path, excludes: list[str], run_tests: bool) -> list[str]:
    """Eksport + czyszczenie ścieżek + skan (+ testy) do `out`. Zwraca listę BŁĘDÓW."""
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

    if run_tests:
        print("\nTesty w wyeksportowanym drzewie (pytest -m 'not slow'):")
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "not slow", "-p", "no:cacheprovider"], cwd=out)
        if r.returncode != 0:
            errors.append(f"testy w wyeksportowanym drzewie nie przechodzą (kod {r.returncode})")
        shutil.rmtree(out / ".pytest_cache", ignore_errors=True)
        for pc in out.rglob("__pycache__"):
            shutil.rmtree(pc, ignore_errors=True)
    return errors


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=cwd, check=check, capture_output=True, text=True)


def _files(root: Path) -> dict[str, Path]:
    return {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts}


def update_target(target: Path, src: Path, author: str | None, message: str | None, dry_run: bool) -> int:
    """Synchronizuje oczyszczone drzewo `src` z klonem `target` i robi w nim zwykły commit."""
    if not (target / ".git").exists():
        print(f"BŁĄD: {target} nie jest klonem git (brak .git)")
        return 1
    if target == ROOT or ROOT in target.parents or target in ROOT.parents:
        print("BŁĄD: klon publiczny nie może być repo źródłowym ani leżeć w nim / je zawierać")
        return 1
    src_remote = _git(ROOT, "remote", "get-url", "origin", check=False).stdout.strip()
    tgt_remote = _git(target, "remote", "get-url", "origin", check=False).stdout.strip()
    if src_remote and src_remote == tgt_remote:
        print("BŁĄD: klon wskazuje na to samo zdalne repo co źródło -- to prywatne archiwum, nie publiczne")
        return 1
    if _git(target, "status", "--porcelain").stdout.strip():
        print("BŁĄD: klon publiczny ma niezacommitowane zmiany lub nieśledzone pliki -- posprzątaj najpierw")
        return 1

    tracked = set(_git(target, "ls-files").stdout.splitlines())
    exported = _files(src)
    added = sorted(set(exported) - tracked)
    removed = sorted(tracked - set(exported))
    modified = sorted(
        p
        for p in set(exported) & tracked
        if (target / p).read_bytes() != exported[p].read_bytes()
        or ((target / p).stat().st_mode & 0o111) != (exported[p].stat().st_mode & 0o111)
    )
    print(f"\nSynchronizacja z {target}: +{len(added)} nowych, ~{len(modified)} zmienionych, -{len(removed)} usuniętych")
    for label, lst in (("nowe", added), ("zmienione", modified), ("usunięte", removed)):
        for p in lst[:30]:
            print(f"  {label}: {p}")
        if len(lst) > 30:
            print(f"  ... i {len(lst) - 30} więcej ({label})")
    if not (added or modified or removed):
        print("Brak zmian -- publiczny klon jest aktualny.")
        return 0
    if dry_run:
        print("(--dry-run: nic nie zapisano)")
        return 0

    for i in range(0, len(removed), 100):
        _git(target, "rm", "-q", "--", *removed[i : i + 100])
    for p in added + modified:
        dst = target / p
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exported[p], dst)
    _git(target, "add", "-A")
    name, mail = re.fullmatch(r"(.+) <(.+)>", author).groups()
    src_sha = _git(ROOT, "rev-parse", "--short", "HEAD").stdout.strip()
    msg = message or f"Aktualizacja z archiwum (wersja {src_sha})"
    _git(target, "-c", f"user.name={name}", "-c", f"user.email={mail}", "commit", "-q", "-m", msg)
    sha = _git(target, "rev-parse", "--short", "HEAD").stdout.strip()
    print(f"\nCommit {sha} w {target}, autor {author}. NIC nie wypchnięto -- po przeglądzie: git -C \"{target}\" push")
    return 0


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
    ap.add_argument("--update", type=Path, default=None, metavar="KLON", help="Zaktualizuj istniejący klon publicznego repo zwykłym commitem")
    ap.add_argument("--message", type=str, default=None, help="Komunikat commitu dla --update")
    args = ap.parse_args()

    if args.update:
        if args.out is not None or args.init_git:
            ap.error("--update nie łączy się z katalogiem wyjściowym ani z --init-git")
        if args.keep_data:
            ap.error("--update nie pozwala na --keep-data (dane zostają w prywatnym archiwum)")
        if not args.dry_run and not args.author:
            ap.error("--update wymaga --author (adres noreply GitHuba), chyba że --dry-run")

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
    if args.update:
        target = args.update.resolve()
        tmp = Path(tempfile.mkdtemp(prefix="export_public_"))
        try:
            errors = build_clean_tree(tmp, excludes, args.run_tests)
            if errors:
                print("\nBŁĘDY -- klon publiczny NIE został zmieniony:")
                for e in errors:
                    print(f"  ! {e}")
                return 1
            return update_target(target, tmp, args.author, args.message, args.dry_run)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if args.dry_run:
        return 0

    if args.out is None:
        ap.error("podaj katalog wyjściowy (albo --dry-run)")
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        ap.error("katalog wyjściowy nie może leżeć wewnątrz repo źródłowego")
    if out.exists() and any(out.iterdir()):
        ap.error(f"{out} nie jest pusty -- nic nie nadpisuję")

    errors = build_clean_tree(out, excludes, args.run_tests)

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
