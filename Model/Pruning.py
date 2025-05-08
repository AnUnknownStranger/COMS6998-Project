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
from preprocess.preprocess import getData, getTest


# -----------------------------------------------------------------------------
# Layerwise Pruning Utility
# -----------------------------------------------------------------------------
def apply_layerwise_pruning(
    model: VisionEncoderDecoderModel,
    *,
    target: Literal['decoder', 'encoder', 'all'] = 'decoder',
    default_amount: float = 0.2,
) -> VisionEncoderDecoderModel:
    """
    Apply L1 unstructured pruning per Linear layer and remove pruning reparam immediately.
    """
    submodules = {'decoder': model.decoder, 'encoder': model.encoder, 'all': model}
    if target not in submodules:
        raise ValueError("target must be 'decoder', 'encoder', or 'all'")
    submod = submodules[target]

    total_params = 0
    total_zeros = 0
    for name, module in submod.named_modules():
        if isinstance(module, torch.nn.Linear):
            if 'lm_head' in name:
                continue
            amount = default_amount
            if 'self_attn' in name:
                amount = 0.1
            elif 'fc2' in name:
                amount = 0.3

            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.remove(module, 'weight')

            num = module.weight.numel()
            zeros = int((module.weight == 0).sum().item())
            total_params += num
            total_zeros += zeros
            print(f"Pruned {name}: sparsity={zeros/num:.2%}")

    overall = (total_zeros / total_params) if total_params > 0 else 0
    print(f"Overall sparsity ({target}): {overall:.2%} ({total_zeros}/{total_params})")
    return model


# -----------------------------------------------------------------------------
# Data Collator for Vision-to-Text
# -----------------------------------------------------------------------------
def build_collator(processor: TrOCRProcessor):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        pixel_values = torch.stack([item['pixel_values'] for item in batch])
        labels = [item['labels'].clone().detach() for item in batch]
        labels_padded = pad_sequence(
            labels,
            batch_first=True,
            padding_value=processor.tokenizer.pad_token_id,
        )
        labels_padded[labels_padded == processor.tokenizer.pad_token_id] = -100
        return {'pixel_values': pixel_values, 'labels': labels_padded}
    return collate_fn


# -----------------------------------------------------------------------------
# Fine-tuning Pipeline
# -----------------------------------------------------------------------------
def fine_tune_model(
    *,
    model_dir: str = 'pruned_model',
    data_dir: str = '/home/hz2994/.cache/kagglehub/datasets/landlord/handwriting-recognition/versions/1',
    epochs: int = 10,
    batch_size: int = 4,
) -> None:
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten', use_fast=True)

    print('[Init] No existing model, downloading and pruning...')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    model = apply_layerwise_pruning(model, target='decoder', default_amount=0.2)
    os.makedirs(model_dir, exist_ok=True)
    model.save_pretrained(model_dir)
    processor.save_pretrained(model_dir)
    print(f'[Init] Pruned model saved to {model_dir}')
    # Check if pruned model can generate predictions
    model.eval()
    dummy_input = torch.randn(1, 3, 384, 384).to(model.device)  # 1 dummy image
    with torch.no_grad():
        try:
            output = model.generate(dummy_input)
            print("✅ Generate successful:", output)
        except Exception as e:
            print("❌ Generate failed after pruning:", e)
            exit(1)


    model.config.decoder_start_token_id = processor.tokenizer.pad_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    train_data = getData('written_name_train_v2.csv', data_dir, 'train_v2/train', processor)
    print(f'[Data] Loaded {len(train_data)} samples')
    collate_fn = build_collator(processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=model_dir,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=1e-5,
        weight_decay=0.01,
        warmup_steps=500,
        save_strategy='epoch',
        save_total_limit=1,
        logging_dir=os.path.join(model_dir, 'logs'),
        remove_unused_columns=False,
        fp16=False,
        max_grad_norm=1.0,
        report_to=['wandb'],
        logging_steps=50,
        run_name='pruned_model',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=processor.tokenizer,
        data_collator=collate_fn,
    )

    start = time.time()
    trainer.train()
    print(f'[Train] Completed in {time.time() - start:.2f}s')

    trainer.save_model(model_dir)
    processor.save_pretrained(model_dir)
    print(f'[Train] Model and processor saved to {model_dir}')


# -----------------------------------------------------------------------------
# Optional: similarity metric
# -----------------------------------------------------------------------------
def calculate_similarity(pred: str, gold: str) -> float:
    lev = distance(pred, gold)
    return 1 - lev / max(len(pred), len(gold)) if max(len(pred), len(gold)) else 1.0


if __name__ == '__main__':
    fine_tune_model()
