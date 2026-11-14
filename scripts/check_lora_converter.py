"""Walidacja end-to-end scripts/convert_lora.py na prawdziwych bibliotekach (PROGRESS.md, Krok 73).

Mały, losowy model o architekturze Bielika (LlamaForCausalLM, GQA) budowany
w transformers; te same wagi bazowe ładowane w mlx_lm. Sprawdza, czy logity
modelu z adapterem są takie same w PEFT i w mlx_lm po konwersji -- czyli
naraz: nazwy kluczy, transpozycje, skalę (alpha/r, rsLoRA), wypełnianie luk
w warstwach i to, że mlx_lm faktycznie nakłada LoRA tam, gdzie PEFT.

1. PEFT -> MLX: adapter PEFT z losowymi A i B (nie zerowe B jak po inicjalizacji),
   wszystkie 7 projekcji, warstwy z luką [1, 3, 4] z 5, alpha/r = 3.
2. MLX -> PEFT: wynik kroku 1 z powrotem do PEFT, PeftModel.from_pretrained.
3. To samo z use_rslora=True (alpha/sqrt(r)).
4. Prawdziwy adapter (domyślnie konsumenckie-distill iter200): MLX -> PEFT -> MLX
   bit w bit (bez ładowania Bielika).

Model liczony na CPU (mx.cpu) -- maleńki, nie koliduje z procesem Bielika
na GPU, ale i tak nie ładuje żadnego dużego modelu.

Wymaga `peft` (nie ma go we wspólnym venv -- np. pip install --target <katalog>
peft accelerate --no-deps i PYTHONPATH=<katalog>).

Użycie:
    PYTHONPATH=<katalog z peft> python scripts/check_lora_converter.py
"""

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import mlx.core as mx  # noqa: E402
from mlx_lm.tuner.utils import load_adapters  # noqa: E402
from mlx_lm.utils import load_model  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

import convert_lora  # noqa: E402

NUM_LAYERS = 5
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
TOL = 1e-4


def tiny_base(path: Path) -> LlamaForCausalLM:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(cfg).float().eval()
    model.save_pretrained(path, safe_serialization=True)
    return model


def torch_logits(model, ids: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(torch.tensor(ids)).logits.float().numpy()


def mlx_logits(base_path: Path, adapter_path: Path | None, ids: np.ndarray) -> np.ndarray:
    model, _ = load_model(base_path)
    model.set_dtype(mx.float32)
    if adapter_path is not None:
        load_adapters(model, str(adapter_path))
    out = model(mx.array(ids))
    return np.array(out.astype(mx.float32))


def compare(name: str, a: np.ndarray, b: np.ndarray, delta: np.ndarray) -> bool:
    diff = float(np.abs(a - b).max())
    effect = float(np.abs(delta).max())
    ok = diff < TOL and effect > 100 * TOL
    print(f"  {'OK ' if ok else 'BŁĄD'} {name}: max|różnica|={diff:.2e} (wpływ adaptera na logity {effect:.2e})")
    return ok


def peft_case(tmp: Path, base: LlamaForCausalLM, ids: np.ndarray, base_t: np.ndarray, use_rslora: bool) -> bool:
    tag = "rslora" if use_rslora else "lora"
    torch.manual_seed(1)
    cfg = LoraConfig(r=4, lora_alpha=12, target_modules=TARGETS, layers_to_transform=[1, 3, 4],
                     lora_dropout=0.0, init_lora_weights=False, use_rslora=use_rslora, task_type="CAUSAL_LM")
    peft_model = get_peft_model(LlamaForCausalLM.from_pretrained(tmp / "base").float().eval(), cfg).eval()
    peft_dir = tmp / f"peft_{tag}"
    peft_model.save_pretrained(peft_dir)
    peft_t = torch_logits(peft_model, ids)

    mlx_dir = tmp / f"mlx_{tag}"
    convert_lora.peft_to_mlx(peft_dir, mlx_dir, NUM_LAYERS)
    ok = compare(f"[{tag}] PEFT -> mlx_lm", peft_t, mlx_logits(tmp / "base", mlx_dir, ids), peft_t - base_t)

    back_dir = tmp / f"peft_back_{tag}"
    convert_lora.mlx_to_peft(mlx_dir, back_dir, "lokalny-test")
    back = PeftModel.from_pretrained(LlamaForCausalLM.from_pretrained(tmp / "base").float().eval(), back_dir).eval()
    ok &= compare(f"[{tag}] mlx_lm -> PEFT", peft_t, torch_logits(back, ids), peft_t - base_t)
    return ok


def real_roundtrip(tmp: Path, adapter: Path) -> bool:
    convert_lora.mlx_to_peft(adapter, tmp / "real_peft", convert_lora.DEFAULT_BASE_MODEL)
    convert_lora.peft_to_mlx(tmp / "real_peft", tmp / "real_mlx", 50)
    a = convert_lora.load_safetensors(adapter / convert_lora.MLX_WEIGHTS)
    b = convert_lora.load_safetensors(tmp / "real_mlx" / convert_lora.MLX_WEIGHTS)
    same = a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)
    print(f"  {'OK ' if same else 'BŁĄD'} prawdziwy adapter {adapter.name}: MLX -> PEFT -> MLX bit w bit ({len(a)} tensorów)")
    return same


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--real-adapter", type=Path,
                        default=ROOT / "adapters" / "bielik11b-konsumenckie-distill-lora-iter200")
    args = parser.parse_args()
    mx.set_default_device(mx.cpu)
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        base = tiny_base(tmp / "base")
        ids = np.random.default_rng(0).integers(0, 97, size=(2, 12))
        base_t = torch_logits(base, ids)
        ok = compare("baza transformers vs mlx_lm (punkt odniesienia)", base_t,
                     mlx_logits(tmp / "base", None, ids), np.full(1, 1.0))
        ok &= peft_case(tmp, base, ids, base_t, use_rslora=False)
        ok &= peft_case(tmp, base, ids, base_t, use_rslora=True)
        if args.real_adapter.exists():
            ok &= real_roundtrip(tmp, args.real_adapter)
    print("WSZYSTKO OK" if ok else "SĄ BŁĘDY")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
