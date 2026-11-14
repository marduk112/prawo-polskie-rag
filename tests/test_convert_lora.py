"""Testy dla scripts/convert_lora.py (konwersja LoRA mlx_lm <-> PEFT, Krok 73).

Bez modelu i bez `peft` -- sama matematyka/klucze/configi na małych tensorach.
Pełna walidacja na prawdziwych bibliotekach: scripts/check_lora_converter.py.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import convert_lora  # noqa: E402


def _mlx_adapter(path: Path, layers=(2, 3), rank=4, scale=20.0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    weights = {}
    for layer in layers:
        weights[f"model.layers.{layer}.self_attn.q_proj.lora_a"] = rng.normal(size=(16, rank)).astype(np.float32)
        weights[f"model.layers.{layer}.self_attn.q_proj.lora_b"] = rng.normal(size=(rank, 16)).astype(np.float32)
        weights[f"model.layers.{layer}.mlp.down_proj.lora_a"] = rng.normal(size=(40, rank)).astype(np.float32)
        weights[f"model.layers.{layer}.mlp.down_proj.lora_b"] = rng.normal(size=(rank, 16)).astype(np.float32)
    convert_lora.save_safetensors(weights, path / convert_lora.MLX_WEIGHTS)
    cfg = {"fine_tune_type": "lora", "num_layers": len(layers),
           "lora_parameters": {"rank": rank, "scale": scale, "dropout": 0.0}}
    (path / convert_lora.CONFIG).write_text(json.dumps(cfg))
    return weights


def test_alpha_scale_roundtrip():
    assert convert_lora.scale_to_alpha(20.0, 8) == 160
    assert convert_lora.alpha_to_scale(160, 8) == 20.0
    assert convert_lora.alpha_to_scale(24, 4, use_rslora=True) == 12.0


def test_mlx_to_peft_keys_transposes_and_config(tmp_path):
    src, dst = tmp_path / "mlx", tmp_path / "peft"
    weights = _mlx_adapter(src)
    convert_lora.mlx_to_peft(src, dst, "base")
    peft = convert_lora.load_safetensors(dst / convert_lora.PEFT_WEIGHTS)
    a = peft["base_model.model.model.layers.2.self_attn.q_proj.lora_A.weight"]
    np.testing.assert_array_equal(a, weights["model.layers.2.self_attn.q_proj.lora_a"].T)
    cfg = json.loads((dst / convert_lora.CONFIG).read_text())
    assert cfg["r"] == 4 and cfg["lora_alpha"] == 80
    assert cfg["layers_to_transform"] == [2, 3]
    assert cfg["target_modules"] == ["mlp.down_proj", "self_attn.q_proj"]


def test_roundtrip_is_exact(tmp_path):
    weights = _mlx_adapter(tmp_path / "mlx", layers=(4, 5))
    convert_lora.mlx_to_peft(tmp_path / "mlx", tmp_path / "peft", "base")
    convert_lora.peft_to_mlx(tmp_path / "peft", tmp_path / "back", num_hidden_layers=6)
    back = convert_lora.load_safetensors(tmp_path / "back" / convert_lora.MLX_WEIGHTS)
    assert back.keys() == weights.keys()
    for k in weights:
        np.testing.assert_array_equal(back[k], weights[k])
    cfg = json.loads((tmp_path / "back" / convert_lora.CONFIG).read_text())
    assert cfg["num_layers"] == 2 and cfg["lora_parameters"]["scale"] == 20.0


def test_layer_gaps_and_trailing_layers_are_zero_filled(tmp_path):
    _mlx_adapter(tmp_path / "mlx", layers=(1, 3))
    convert_lora.mlx_to_peft(tmp_path / "mlx", tmp_path / "peft", "base")
    convert_lora.peft_to_mlx(tmp_path / "peft", tmp_path / "back", num_hidden_layers=5)
    back = convert_lora.load_safetensors(tmp_path / "back" / convert_lora.MLX_WEIGHTS)
    cfg = json.loads((tmp_path / "back" / convert_lora.CONFIG).read_text())
    assert cfg["num_layers"] == 4  # warstwy 1..4 -- mlx_lm liczy od końca
    for layer in (2, 4):
        assert not back[f"model.layers.{layer}.self_attn.q_proj.lora_a"].any()
        assert back[f"model.layers.{layer}.mlp.down_proj.lora_b"].shape == (4, 16)


def test_delta_matches_between_formulas(tmp_path):
    """mlx: scale * x@a@b  ==  PEFT: alpha/r * x@A.T@B.T"""
    weights = _mlx_adapter(tmp_path / "mlx", layers=(0,), rank=4, scale=3.0)
    convert_lora.mlx_to_peft(tmp_path / "mlx", tmp_path / "peft", "base")
    peft = convert_lora.load_safetensors(tmp_path / "peft" / convert_lora.PEFT_WEIGHTS)
    cfg = json.loads((tmp_path / "peft" / convert_lora.CONFIG).read_text())
    x = np.random.default_rng(1).normal(size=(3, 16)).astype(np.float32)
    a, b = weights["model.layers.0.self_attn.q_proj.lora_a"], weights["model.layers.0.self_attn.q_proj.lora_b"]
    A = peft["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"]
    B = peft["base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight"]
    np.testing.assert_allclose(3.0 * x @ a @ b, cfg["lora_alpha"] / cfg["r"] * x @ A.T @ B.T, rtol=1e-5)


@pytest.mark.parametrize("field,value", [("use_dora", True), ("rank_pattern", {"q_proj": 16}),
                                         ("modules_to_save", ["lm_head"]), ("bias", "all")])
def test_peft_to_mlx_rejects_unsupported(tmp_path, field, value):
    _mlx_adapter(tmp_path / "mlx")
    convert_lora.mlx_to_peft(tmp_path / "mlx", tmp_path / "peft", "base")
    cfg_path = tmp_path / "peft" / convert_lora.CONFIG
    cfg = json.loads(cfg_path.read_text())
    cfg[field] = value
    cfg_path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        convert_lora.peft_to_mlx(tmp_path / "peft", tmp_path / "back", num_hidden_layers=6)


def test_mlx_to_peft_rejects_unknown_keys(tmp_path):
    _mlx_adapter(tmp_path / "mlx")
    convert_lora.save_safetensors({"model.embed_tokens.lora_a": np.zeros((4, 4), np.float32)},
                                  tmp_path / "mlx" / convert_lora.MLX_WEIGHTS)
    with pytest.raises(ValueError):
        convert_lora.mlx_to_peft(tmp_path / "mlx", tmp_path / "peft", "base")
