import subprocess
import os
import time
import numpy as np
import torch
from tqdm import tqdm
import wandb

from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from preprocess.preprocess import getTest

def predict_with_dynamic_quant(
    data_dir, csv_filename="written_name_test_v2.csv",
    split_dir="test_v2/test", batch_size=2
):
    """Load model, apply dynamic quantization, and evaluate."""
    torch.cuda.empty_cache()
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("default_model")

    # Quantize Linear layers to INT8
    model.to("cpu")
    model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    print("Model quantized to INT8.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Load test data
    print("Loading test data...")
    test_dataset = getTest(csv_filename, data_dir, split_dir, processor)

    run = wandb.init(project="Trocr", name="dynamic-quant-eval", reinit=True)
    preds, labels = [], []

    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch = test_dataset[i : i + batch_size]
        images = torch.stack([x["pixel_values"] for x in batch]).float().to(device)

        with torch.no_grad():
            outputs = model.generate(images)
        decoded = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, out in enumerate(outputs):
            pred = decoded[j]
            label_ids = [l for l in batch[j]["labels"] if l != -100]
            actual = processor.tokenizer.decode(label_ids, skip_special_tokens=True)

            preds.append(pred)
            labels.append(actual)

            if len(preds) <= 50:
                run.log({"id": i + j, "pred": pred, "actual": actual})

        torch.cuda.empty_cache()

    acc = (np.array(preds) == np.array(labels)).mean()
    run.log({"accuracy": acc})
    run.finish()

    print(f"Accuracy: {acc:.4f}")

if __name__ == "__main__":
    # Train model via Default.py
    subprocess.run(["python", "./Model/Default.py"], check=True)
    # Run quantized inference
    predict_with_dynamic_quant(
        data_dir="/home/tt3010/COMS6998-Project/dataset"
    )
