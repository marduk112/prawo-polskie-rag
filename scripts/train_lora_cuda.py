"""
Przetestowane end-to-end na realnej karcie NVIDIA (RunPod, A40, 48GB
VRAM) -- patrz PROGRESS.md, Krok 26. Dwa bugi znalezione i naprawione
przy tej okazji: brak `processing_class` w `SFTTrainer` (parametr
`tokenizer` został usunięty z API `trl`) oraz niekompatybilny z
`assistant_only_loss` chat_template Bielika (ChatML sklejony w jednym
bloku Jinja -- wymaga ręcznego dopisania `{% generation %}`, `trl` nie
potrafi go auto-załatać). Zalecany model/adapter: 11B, patrz Krok 26 --
4.5B ma udokumentowaną (Krok 16, Krok 25, Krok 26) tendencję do
przesłaniania kontekstu z RAG pewną, ale zmyśloną liczbą z wag.

Odpowiednik `mlx_lm.lora` (patrz PROGRESS.md, kroki 4b i 6) dla
Linuksa/Windows z kartą NVIDIA: LoRA fine-tuning Bielika przez
transformers + peft + bitsandbytes (kwantyzacja 4-bit) + trl
(SFTTrainer). Trenuje na tych samych plikach co wariant MLX
(data/finetune/train.jsonl, valid.jsonl, format {"messages": [...]}).

Domyślne hiperparametry (batch 2, 200 kroków, ewaluacja/zapis co 25,
lr 1e-5, maskowanie promptu w loss) startowo przeniesione 1:1 z
przebiegów na MLX; od Kroku 73 także zakres LoRA (7 projekcji, ostatnie
16 warstw, alpha 160 == scale 20 przy r=8) i stały LR -- adapter da się
przenieść na Maca przez scripts/convert_lora.py peft-to-mlx. Historycznie
`SFTConfig` domyślnie ustawia
`lr_scheduler_type="linear"` (LR opada do ~0 przez cały trening), a
`mlx_lm.lora` nie ma żadnego schedulera (LR stały przez wszystkie
iteracje). W testach (Krok 26) TA różnica -- nie sam framework --
tłumaczy, dlaczego tu (w przeciwieństwie do MLX) `eval_loss` maleje
monotonicznie i się wypłaszcza zamiast rosnąć po znalezieniu minimum:
`--lr-scheduler-type constant` odtwarza bliżej dynamikę MLX (przydatne
do porównań), ale przy dłuższym treningu i tak zaleca się obejrzeć
`eval_loss` w logach, nie ufać ślepo ostatniemu krokowi (patrz
--save-steps -- każdy zapis trafia do osobnego podkatalogu
checkpoint-N w --output-dir).

Użycie:
    python scripts/train_lora_cuda.py

    python scripts/train_lora_cuda.py \\
        --model speakleash/Bielik-4.5B-v3.0-Instruct \\
        --output-dir adapters-cuda/bielik4.5b-kadry-lora
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "finetune"
DEFAULT_MODEL = "speakleash/Bielik-11B-v3.0-Instruct"
DEFAULT_OUTPUT_DIR = ROOT / "adapters-cuda" / "bielik11b-kadry-lora-hedge"


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning Bielika na CUDA (transformers + peft + trl)")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Repo Hugging Face modelu bazowego")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Katalog z train.jsonl / valid.jsonl")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    # Krok 73: domyślne wyrównane z mlx_lm.lora (adaptery w adapters/), żeby adapter
    # z CUDA po scripts/convert_lora.py peft-to-mlx był odpowiednikiem lokalnego:
    # scale 20 w mlx_lm == lora_alpha / r -> alpha = 20 * 8 = 160; LoRA na
    # wszystkich 7 projekcjach w ostatnich 16 warstwach.
    parser.add_argument("--lora-alpha", type=int, default=160)
    parser.add_argument(
        "--num-layers",
        type=int,
        default=16,
        help="LoRA na ostatnich N warstwach (jak --num-layers w mlx_lm.lora); <= 0 = wszystkie",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="constant",
        help="'constant' jak mlx_lm.lora (stały LR przez wszystkie iteracje); 'linear' (domyślny "
        "SFTConfig) maleje do ~0 przez cały trening",
    )
    # Krok 76: SFTConfig.max_length domyślnie 1024 i obcina od końca
    # (truncation_mode keep_start) -- przykłady destylacji (prompt z kontekstem
    # RAG, 8-11 tys. tokenów) traciłyby całą odpowiedź asystenta, czyli loss
    # liczyłby się z niczego. 11264 jak max_seq_length w mlx_lm (Krok 72).
    parser.add_argument("--max-length", type=int, default=11264)
    parser.add_argument(
        "--quantization",
        choices=["4bit", "none"],
        default="4bit",
        help="'none' = bazowy model w bf16 (szybciej, ~22 GB dla 11B -- karta 48 GB+)",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=0,
        help="Walidacja na N pierwszych przykładach valid.jsonl (0 = wszystkie) -- przy długich "
        "przykładach pełna walidacja co eval-steps kosztuje tyle co dziesiątki kroków treningu",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        choices=["on", "off"],
        default="on",
        help="'off' (tylko z --quantization none): bez ponownego liczenia aktywacji i bez gradientu "
        "przez zamrożone dolne warstwy -- szybciej, więcej pamięci (Krok 78)",
    )
    args = parser.parse_args()
    checkpointing = args.gradient_checkpointing == "on" or args.quantization == "4bit"

    data_dir = Path(args.data_dir)
    print(f"[INFO] Ładowanie modelu bazowego ({args.model}), kwantyzacja: {args.quantization}...")
    if args.quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb_config, device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Bielik's stock chat template (ChatML: <|im_start|>role\n...<|im_end|>\n) ma
    # role/content sklejone w jednym {{ }} bloku, więc trl nie potrafi go
    # automatycznie załatać markerami {% generation %} wymaganymi przez
    # assistant_only_loss (patrz ValueError "chat template is not
    # training-compatible"). Wersja niżej jest funkcjonalnie identyczna --
    # renderuje ten sam tekst -- tylko z turą asystenta owiniętą w
    # {% generation %}, żeby loss liczył się wyłącznie na jej tokenach.
    tokenizer.chat_template = (
        "{{bos_token}}{% for message in messages %}"
        "{% if message['role'] == 'assistant' %}"
        "{{'<|im_start|>' + message['role'] + '\n'}}"
        "{% generation %}{{message['content'] + '<|im_end|>' + '\n'}}{% endgeneration %}"
        "{% else %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )

    n_layers = model.config.num_hidden_layers
    layers = list(range(max(n_layers - args.num_layers, 0), n_layers)) if args.num_layers > 0 else None
    if args.quantization == "4bit":
        model = prepare_model_for_kbit_training(model)
    else:
        if checkpointing:
            model.enable_input_require_grads()  # gradient checkpointing przy zamrożonych embeddingach
        # Bez checkpointingu wejścia nie wymagają gradientu, więc autograd liczy wstecz tylko
        # przez warstwy z LoRA (ostatnie --num-layers), nie przez wszystkie.
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        # jak mlx_lm.lora: wszystkie liniowe projekcje bloku, nie tylko q/v
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        layers_to_transform=layers,
        layers_pattern="layers" if layers is not None else None,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"[INFO] Wczytywanie danych z {data_dir}...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(data_dir / "train.jsonl"),
            "validation": str(data_dir / "valid.jsonl"),
        },
    )
    if args.max_eval_samples:
        dataset["validation"] = dataset["validation"].select(range(min(args.max_eval_samples, len(dataset["validation"]))))

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        logging_steps=10,
        max_length=args.max_length,
        per_device_eval_batch_size=1,
        gradient_checkpointing=checkpointing,
        assistant_only_loss=True,  # odpowiednik --mask-prompt w mlx_lm.lora
        bf16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )
    trainer.train()
    if torch.cuda.is_available():
        print(f"[INFO] Szczyt pamięci GPU (PyTorch): {torch.cuda.max_memory_reserved() / 2**30:.1f} GiB")

    final_path = Path(args.output_dir) / "final"
    trainer.save_model(str(final_path))
    print(f"[INFO] Zapisano finalny adapter w {final_path}")
    print(
        "[INFO] Sprawdź eval_loss w logach powyżej i porównaj z checkpointami w "
        f"{args.output_dir}/checkpoint-* -- w naszych przebiegach na MLX najlepszy "
        "checkpoint NIE był ostatnim (patrz PROGRESS.md)."
    )


if __name__ == "__main__":
    main()
