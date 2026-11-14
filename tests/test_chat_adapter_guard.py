"""Krok 118: domyślne adaptery LoRA (trenowane na Bieliku v3.0) nie mogą ładować się na innej bazie (np. v3.1 8-bit)."""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

# chat.py importuje MLX na poziomie modułu (tylko Apple Silicon). Testowana logika (`resolve_adapter_path`)
# go nie używa, więc tam, gdzie MLX nie ma (Linux w CI), podstawiamy atrapy -- test biegnie wszędzie, nie jest pomijany.
if importlib.util.find_spec("mlx") is None:
    for name in ("mlx", "mlx.utils", "mlx_lm", "mlx_lm.tuner", "mlx_lm.tuner.lora"):
        sys.modules.setdefault(name, MagicMock())

import chat  # noqa: E402

V30 = "models/Bielik-11B-v3.0-Instruct-mlx"
V31 = "models/Bielik-11B-v3.1-Instruct-mlx-8bit"
DOMAIN = next(iter(chat.DEFAULT_ADAPTER_PATHS))


def test_default_adapter_loaded_for_v30():
    assert chat.resolve_adapter_path(V30, DOMAIN) == str(chat.DEFAULT_ADAPTER_PATHS[DOMAIN])


def test_default_adapter_skipped_for_v31():
    assert chat.resolve_adapter_path(V31, DOMAIN) is None


def test_explicit_adapter_always_wins_even_on_v31():
    assert chat.resolve_adapter_path(V31, DOMAIN, explicit_adapter="adapters/x") == "adapters/x"


def test_no_adapter_flag_wins_over_explicit_and_default():
    assert chat.resolve_adapter_path(V30, DOMAIN, no_adapter=True, explicit_adapter="adapters/x") is None


def test_extra_domain_from_router_disables_default_adapter():
    assert chat.resolve_adapter_path(V30, DOMAIN, has_extra_domains=True) is None


def test_domain_without_default_adapter_gets_none():
    assert chat.resolve_adapter_path(V30, "domena_bez_adaptera") is None


def test_answer_model_is_v31_8bit_and_not_adapter_base():
    assert chat.ANSWER_MODEL_PATH.name == "Bielik-11B-v3.1-Instruct-mlx-8bit"
    assert not chat.ANSWER_MODEL_PATH.name.startswith(chat.ADAPTER_BASE_MODEL_PREFIX)


def test_model_path_and_expansion_model_stay_on_v30():
    # Inne skrypty importują MODEL_PATH jako bazę adapterów/ekspansji -- zmiana to regresja retrievalu (Krok 118).
    assert chat.MODEL_PATH.name.startswith(chat.ADAPTER_BASE_MODEL_PREFIX)
    assert chat.EXPAND_MODEL_PATH == chat.MODEL_PATH
