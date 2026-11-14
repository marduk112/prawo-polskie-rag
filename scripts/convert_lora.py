"""Konwersja adapterów LoRA między mlx_lm (Mac) a PEFT (CUDA) -- PROGRESS.md, Krok 73.

Po co: trening na CUDA (RunPod, scripts/train_lora_cuda.py) jest wielokrotnie
tańszy w czasie niż lokalny przy długich przykładach z kontekstem RAG (Krok 72:
~77 s/iter na M4 Pro), a produkcyjnie adaptery działają na Macu (chat.py, MLX).
Kierunek mlx-to-peft pozwala z kolei ocenić / dotrenować lokalne adaptery na CUDA.

Matematyka (obie biblioteki, bez dropoutu w inferencji):
    mlx_lm:  y = W x + scale * (x @ lora_a) @ lora_b      lora_a: (in, r), lora_b: (r, out)
    PEFT:    y = W x + (alpha / r) * B(A(x))              A.weight: (r, in), B.weight: (out, r)
             (rsLoRA: alpha / sqrt(r))
=> A = lora_a.T, B = lora_b.T, scale = alpha / r (alpha / sqrt(r) przy use_rslora).

Klucze: mlx `model.layers.N.self_attn.q_proj.lora_a` <->
PEFT `base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight`.

mlx_lm nakłada LoRA na OSTATNIE `num_layers` warstw; jeśli adapter PEFT
obejmuje inne warstwy (np. wszystkie albo z lukami), num_layers liczone jest
od najniższej warstwy adaptera do końca modelu, a warstwy bez wag w PEFT
dostają zerowe lora_a/lora_b (wkład dokładnie 0). Moduły ograniczane przez
`lora_parameters.keys`.

Nieobsługiwane (błąd zamiast cichej, złej konwersji): DoRA, rank_pattern /
alpha_pattern, bias != "none", modules_to_save, LoRA na embeddingach.

Użycie:
    python scripts/convert_lora.py mlx-to-peft adapters/bielik11b-konsumenckie-distill-lora-iter200 \\
        adapters-cuda/bielik11b-konsumenckie-distill-lora-iter200
    python scripts/convert_lora.py peft-to-mlx adapters-cuda/<adapter> adapters/<adapter> \\
        --num-hidden-layers 50
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

MLX_WEIGHTS = "adapters.safetensors"
PEFT_WEIGHTS = "adapter_model.safetensors"
CONFIG = "adapter_config.json"
PEFT_PREFIX = "base_model.model."
DEFAULT_BASE_MODEL = "speakleash/Bielik-11B-v3.0-Instruct"

MLX_KEY_RE = re.compile(r"^(model\.layers\.(\d+)\.(.+))\.lora_([ab])$")
PEFT_KEY_RE = re.compile(r"^base_model\.model\.(model\.layers\.(\d+)\.(.+))\.lora_([AB])\.weight$")


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    # przez torch, bo safetensors.numpy nie czyta bf16 (PEFT często zapisuje bf16)
    from safetensors.torch import load_file

    return {k: v.float().numpy() for k, v in load_file(str(path)).items()}


def save_safetensors(tensors: dict[str, np.ndarray], path: Path, metadata: dict | None = None) -> None:
    from safetensors.numpy import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: np.ascontiguousarray(v) for k, v in tensors.items()}, str(path), metadata=metadata)


def scale_to_alpha(scale: float, rank: int, use_rslora: bool = False) -> float:
    alpha = scale * (math.sqrt(rank) if use_rslora else rank)
    return int(alpha) if float(alpha).is_integer() else alpha


def alpha_to_scale(alpha: float, rank: int, use_rslora: bool = False) -> float:
    return alpha / (math.sqrt(rank) if use_rslora else rank)


def mlx_to_peft_tensors(weights: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], list[int], list[str]]:
    out, layers, modules = {}, set(), set()
    for key, value in weights.items():
        m = MLX_KEY_RE.match(key)
        if not m:
            raise ValueError(f"Nieobsługiwany klucz adaptera MLX (np. DoRA/embedding?): {key}")
        path, layer, module, ab = m.groups()
        out[f"{PEFT_PREFIX}{path}.lora_{ab.upper()}.weight"] = value.T
        layers.add(int(layer))
        modules.add(module)
    return out, sorted(layers), sorted(modules)


def peft_to_mlx_tensors(
    weights: dict[str, np.ndarray], num_hidden_layers: int
) -> tuple[dict[str, np.ndarray], int, list[str]]:
    out, layers, shapes = {}, set(), {}
    for key, value in weights.items():
        m = PEFT_KEY_RE.match(key)
        if not m:
            raise ValueError(f"Nieobsługiwany klucz adaptera PEFT (modules_to_save/embedding/bias?): {key}")
        path, layer, module, ab = m.groups()
        out[f"{path}.lora_{ab.lower()}"] = value.T
        layers.add(int(layer))
        shapes[(module, ab.lower())] = value.T.shape
    modules = sorted({module for module, _ in shapes})
    first = min(layers)
    if max(layers) >= num_hidden_layers:
        raise ValueError(f"Adapter ma warstwę {max(layers)}, a model tylko {num_hidden_layers}")
    # mlx_lm obejmuje ostatnie num_layers warstw -- brakujące dostają zera (wkład 0,
    # zamiast losowej inicjalizacji lora_a z linear_to_lora_layers)
    for layer in range(first, num_hidden_layers):
        for module in modules:
            for ab in "ab":
                key = f"model.layers.{layer}.{module}.lora_{ab}"
                if key not in out:
                    out[key] = np.zeros(shapes[(module, ab)], dtype=np.float32)
    return out, num_hidden_layers - first, modules


def check_rank(tensors: dict[str, np.ndarray], rank: int, kind: str) -> None:
    for key, value in tensors.items():
        if rank not in value.shape:
            raise ValueError(f"{kind}: {key} ma kształt {value.shape}, oczekiwany rank {rank} (rank_pattern?)")


def mlx_to_peft(src: Path, dst: Path, base_model: str) -> None:
    cfg = json.loads((src / CONFIG).read_text(encoding="utf-8"))
    if cfg.get("fine_tune_type", "lora") != "lora":
        raise ValueError(f"Obsługiwany tylko fine_tune_type=lora, jest {cfg.get('fine_tune_type')}")
    params = cfg["lora_parameters"]
    weights = load_safetensors(src / MLX_WEIGHTS)
    check_rank(weights, params["rank"], "MLX")
    tensors, layers, modules = mlx_to_peft_tensors(weights)
    peft_cfg = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": base_model,
        "r": params["rank"],
        "lora_alpha": scale_to_alpha(params["scale"], params["rank"]),
        "lora_dropout": params.get("dropout", 0.0),
        "use_rslora": False,
        "use_dora": False,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        # PEFT dopasowuje target_modules po końcówce nazwy -- pełne ścieżki
        # (self_attn.q_proj) są jednoznaczne
        "target_modules": modules,
        "layers_to_transform": layers,
        "layers_pattern": "layers",
    }
    save_safetensors(tensors, dst / PEFT_WEIGHTS, metadata={"format": "pt"})
    (dst / CONFIG).write_text(json.dumps(peft_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[mlx->peft] {len(tensors)} tensorów, warstwy {layers[0]}-{layers[-1]}, moduły {modules}, "
          f"r={peft_cfg['r']}, alpha={peft_cfg['lora_alpha']} -> {dst}")


def peft_to_mlx(src: Path, dst: Path, num_hidden_layers: int) -> None:
    cfg = json.loads((src / CONFIG).read_text(encoding="utf-8"))
    if cfg.get("peft_type", "LORA") != "LORA":
        raise ValueError(f"Obsługiwany tylko peft_type=LORA, jest {cfg.get('peft_type')}")
    for field in ("use_dora", "rank_pattern", "alpha_pattern", "modules_to_save"):
        if cfg.get(field):
            raise ValueError(f"Nieobsługiwane w konwersji do mlx_lm: {field}={cfg[field]!r}")
    if cfg.get("bias", "none") != "none":
        raise ValueError(f"Nieobsługiwane bias={cfg['bias']!r}")
    if cfg.get("fan_in_fan_out"):
        raise ValueError("fan_in_fan_out=True (warstwy Conv1D, np. GPT-2) -- nie dotyczy Bielika, nieobsługiwane")
    rank = cfg["r"]
    weights = load_safetensors(src / PEFT_WEIGHTS)
    check_rank(weights, rank, "PEFT")
    tensors, num_layers, modules = peft_to_mlx_tensors(weights, num_hidden_layers)
    scale = alpha_to_scale(cfg["lora_alpha"], rank, cfg.get("use_rslora", False))
    mlx_cfg = {
        "fine_tune_type": "lora",
        "num_layers": num_layers,
        "lora_parameters": {"rank": rank, "scale": scale, "dropout": cfg.get("lora_dropout", 0.0), "keys": modules},
        "converted_from": str(src),
    }
    save_safetensors(tensors, dst / MLX_WEIGHTS)
    (dst / CONFIG).write_text(json.dumps(mlx_cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[peft->mlx] {len(tensors)} tensorów, num_layers={num_layers}, moduły {modules}, "
          f"rank={rank}, scale={scale} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("mlx-to-peft")
    a.add_argument("src", type=Path)
    a.add_argument("dst", type=Path)
    a.add_argument("--base-model", default=DEFAULT_BASE_MODEL, help="base_model_name_or_path w configu PEFT")
    b = sub.add_parser("peft-to-mlx")
    b.add_argument("src", type=Path)
    b.add_argument("dst", type=Path)
    b.add_argument("--num-hidden-layers", type=int, default=50, help="Liczba warstw modelu bazowego (Bielik 11B: 50)")
    args = parser.parse_args()
    if args.dst.exists() and any(args.dst.iterdir()):
        raise SystemExit(f"Katalog docelowy {args.dst} istnieje i nie jest pusty -- nie nadpisuję")
    if args.cmd == "mlx-to-peft":
        mlx_to_peft(args.src, args.dst, args.base_model)
    else:
        peft_to_mlx(args.src, args.dst, args.num_hidden_layers)


if __name__ == "__main__":
    main()
