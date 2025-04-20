import subprocess
import os
import time
import numpy as np
import torch
from tqdm import tqdm
import wandb
from Levenshtein import distance

from transformers import VisionEncoderDecoderModel, TrOCRProcessor
from preprocess.preprocess import getTest

def calculate_similarity(pred, actual):
    # Calculate Levenshtein distance
    lev_distance = distance(pred, actual)
    # Calculate maximum possible distance (length of longer string)
    max_len = max(len(pred), len(actual))
    # Calculate similarity score (1 - normalized distance)
    similarity = 1 - (lev_distance / max_len) if max_len > 0 else 1.0
    return similarity

def predict_with_dynamic_quant(
    data_dir, csv_filename="written_name_test_v2.csv",
    split_dir="test_v2/test", batch_size=2
):
    """Load model, apply dynamic quantization, and evaluate."""
    torch.cuda.empty_cache()
    
    # Record model loading time
    load_start_time = time.time()
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    # model = VisionEncoderDecoderModel.from_pretrained("/home/tt3010/COMS6998-Project/model/checkpoint-500")
    model = VisionEncoderDecoderModel.from_pretrained("default_model")
    
    # Move model to CPU for quantization
    model = model.to('cpu')
    # Apply dynamic quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    # Keep quantized model on CPU
    device = "cpu"
    quantized_model = quantized_model.to(device)
    load_end_time = time.time()
    load_time = load_end_time - load_start_time
    print(f"Model loading and quantization time: {load_time:.4f} seconds")

    quantized_model.eval()

    # Load test data
    print("Loading test data...")
    test_dataset = getTest(csv_filename, data_dir, split_dir, processor)

    run = wandb.init(project="Trocr", name="dynamic-quant-eval", reinit=True)
    preds, labels, similarities = [], [], []
    
    # Record total inference time
    total_inference_time = 0
    total_samples = 0
    
    inference_start_time = time.time()

    for i in tqdm(range(0, len(test_dataset), batch_size)):
        batch = test_dataset[i : i + batch_size]
        images = torch.stack([x["pixel_values"] for x in batch]).float().to(device)
        
        # Record inference time for each batch
        batch_start_time = time.time()
        with torch.no_grad():
            outputs = quantized_model.generate(images)
        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        
        batch_size_actual = len(batch)
        total_samples += batch_size_actual
        total_inference_time += batch_time
        
        print(f"Batch {i//batch_size} inference time: {batch_time:.4f} seconds, {batch_time/batch_size_actual:.4f} seconds per sample")
        
        decoded = processor.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        for j, out in enumerate(outputs):
            pred = decoded[j]
            label_ids = [l for l in batch[j]["labels"] if l != -100]
            actual = processor.tokenizer.decode(label_ids, skip_special_tokens=True)

            preds.append(pred)
            labels.append(actual)
            
            # Calculate similarity score
            similarity = calculate_similarity(pred, actual)
            similarities.append(similarity)

            if len(preds) <= 50:
                run.log({
                    "id": i + j,
                    "pred": pred,
                    "actual": actual,
                    "similarity": similarity
                })

        torch.cuda.empty_cache()
    
    inference_end_time = time.time()
    total_time = inference_end_time - inference_start_time
    
    print(f"Total inference time: {total_time:.4f} seconds")
    print(f"Average inference time per sample: {total_inference_time/total_samples:.4f} seconds")
    print(f"Processing overhead time: {total_time - total_inference_time:.4f} seconds")

    # Calculate metrics
    preds_np = np.array(preds)
    labels_np = np.array(labels)
    similarities_np = np.array(similarities)
    
    # Exact match accuracy (original metric)
    exact_match = preds_np == labels_np
    accuracy = np.mean(exact_match)
    
    # Average similarity score
    avg_similarity = np.mean(similarities_np)
    
    # Percentage of predictions with similarity > 0.8
    high_similarity = np.mean(similarities_np > 0.8)

    run.log({
        "Test Accuracy (Exact Match)": accuracy,
        "Average Similarity": avg_similarity,
        "High Similarity (>0.8)": high_similarity,
        "total_inference_time": total_time,
        "avg_inference_time": total_inference_time/total_samples
    })
    
    print(f"Exact Match Accuracy: {accuracy:.4f}")
    print(f"Average Similarity Score: {avg_similarity:.4f}")
    print(f"Percentage of High Similarity (>0.8): {high_similarity:.4f}")

    run.finish()

if __name__ == "__main__":
    # Train model via Default.py
    # subprocess.run(["python", "./Model/Default.py"], check=True)
    # Run quantized inference
    predict_with_dynamic_quant(
        data_dir="/home/tt3010/COMS6998-Project/dataset"
    )
