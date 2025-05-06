import os
import time
from typing import Literal, List, Dict, Any

import torch
import torch.nn.utils.prune as prune
from Levenshtein import distance
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from torch.nn.utils.rnn import pad_sequence

from preprocess.preprocess import getData

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def _freeze_zero_weights(module: torch.nn.Module):
    """Freeze parameters that became zero after pruning so gradients stay 0."""
    with torch.no_grad():
        mask = module.weight == 0
        module.weight[mask] = 0
    module.weight.register_hook(lambda grad: grad * (~mask))


def apply_layerwise_pruning(
    model: VisionEncoderDecoderModel,
    *,
    target: Literal["decoder", "encoder", "all"] = "decoder",
    default_amount: float = 0.2,
    freeze_pruned: bool = True,
):
    """Apply Layer‑wise L1‑unstructured pruning and (optionally) freeze pruned params."""
    if target not in {"decoder", "encoder", "all"}:
        raise ValueError("target must be 'decoder', 'encoder', or 'all'")

    submodule = {
        "decoder": model.decoder,
        "encoder": model.encoder,
        "all": model,
    }[target]

    print(f"[Pruning] {target} | base ratio={default_amount}")

    total_params, total_zeros = 0, 0
    for name, module in submodule.named_modules():
        if isinstance(module, torch.nn.Linear):
            # custom ratios
            if "self_attn" in name:
                amount = 0.10
            elif "fc2" in name:
                amount = 0.30
            else:
                amount = default_amount

            prune.l1_unstructured(module, name="weight", amount=amount)
            # keep mask during training
            if freeze_pruned:
                _freeze_zero_weights(module)

            param = module.weight
            total_params += param.numel()
            zeros = torch.sum(param == 0).item()
            total_zeros += zeros
            print(f"  · {name:50s}  sparsity={zeros/param.numel():.2%}")

    print(f"[Pruning] Overall sparsity: {total_zeros/total_params:.2%}\n")
    return model


# -----------------------------------------------------------------------------
# Custom collator for vision‑to‑text
# -----------------------------------------------------------------------------

def build_collator(processor: TrOCRProcessor):
    """Return a collate_fn that pads pixel_values & labels appropriately."""

    def collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([item["pixel_values"] for item in batch])  # images already 3×384×384

        # labels: list[List[int]] ➜ pad to max_len with pad_token_id, then replace pad with -100
        label_seqs = [torch.tensor(item["labels"], dtype=torch.long) for item in batch]
        labels = pad_sequence(
            label_seqs,
            batch_first=True,
            padding_value=processor.tokenizer.pad_token_id,
        )
        labels[labels == processor.tokenizer.pad_token_id] = -100

        return {"pixel_values": pixel_values, "labels": labels}

    return collate


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def fine_tune_model(
    *,
    model_dir: str = "pruned_model",
    data_dir: str = "/home/hz2994/.cache/kagglehub/datasets/landlord/handwriting-recognition/versions/1",
    epochs: int = 10,
    batch_size: int = 4,
):
    """Load (or create) pruned model, then fine‑tune."""
    csv_filename = "written_name_train_v2.csv"
    type_fn = "train_v2/train"

    print("[Init] Loading processor …")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=True)

    if os.path.exists(model_dir):
        print(f"[Init] Using existing model from '{model_dir}' …")
        model = VisionEncoderDecoderModel.from_pretrained(model_dir)
    else:
        print("[Init] Creating pruned model …")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        model = apply_layerwise_pruning(model, target="decoder", default_amount=0.2)
        os.makedirs(model_dir, exist_ok=True)
        model.save_pretrained(model_dir)
        processor.save_pretrained(model_dir)
        print("[Init] Pruned model saved.")

    model.config.decoder_start_token_id = processor.tokenizer.pad_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Data --------------------------------------------------------------------
    print("[Data] Loading dataset …")
    train_data = getData(csv_filename, data_dir, type_fn, processor)
    print(f"[Data] Samples: {len(train_data)}")

    data_collator = build_collator(processor)

    # Trainer -----------------------------------------------------------------
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        save_total_limit=1,
        logging_dir=os.path.join(model_dir, "logs"),
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        learning_rate=1e-5,
        report_to=["wandb"],
        logging_steps=50,
        run_name="pruned_model",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=processor.tokenizer,  # still needed for decoding & metrics
        data_collator=data_collator,
    )

    # Train -------------------------------------------------------------------
    start = time.time()
    trainer.train()
    print(f"[Train] Done in {time.time() - start:.2f}s")

    trainer.save_model(model_dir)
    processor.save_pretrained(model_dir)
    print("[Train] Model saved to", model_dir)


# -----------------------------------------------------------------------------
# Levenshtein helper (optional evaluation)
# -----------------------------------------------------------------------------

def calculate_similarity(pred: str, gold: str) -> float:
    lev = distance(pred, gold)
    return 1 - lev / max(len(pred), len(gold)) if max(len(pred), len(gold)) else 1.0


if __name__ == "__main__":
    fine_tune_model()
