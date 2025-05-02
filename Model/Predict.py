import torch
from transformers import VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator,TrOCRProcessor
import wandb
import os
from preprocess.preprocess import getTest
import time
from tqdm import tqdm
import numpy as np
import sys 
from Levenshtein import distance

def calculate_similarity(pred, actual):
    # Calculate Levenshtein distance
    lev_distance = distance(pred, actual)
    # Calculate maximum possible distance (length of longer string)
    max_len = max(len(pred), len(actual))
    # Calculate similarity score (1 - normalized distance)
    similarity = 1 - (lev_distance / max_len) if max_len > 0 else 1.0
    return similarity

if __name__ == "__main__":

    torch.cuda.empty_cache()
    dir = "/home/wei1070580217/.cache/kagglehub/datasets/landlord/handwriting-recognition/versions/1"
    
    csv_filename = "written_name_test_v2.csv"
    type_fn = "test_v2/test" 

    #Make prediction based on processors
    if len(sys.argv) < 2:
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        model = VisionEncoderDecoderModel.from_pretrained("default_model")
    else:
        type = str(sys.argv[1])
        if type == 'default':
            processor = TrOCRProcessor.from_pretrained("Compile_model_default")
            model = VisionEncoderDecoderModel.from_pretrained("Compile_model_default")
        if type == 'ma':
            processor = TrOCRProcessor.from_pretrained("Compile_model_ma")
            model = VisionEncoderDecoderModel.from_pretrained("Compile_model_ma")
        if type == 'ro':
            processor = TrOCRProcessor.from_pretrained("Compile_model_ro")
            model = VisionEncoderDecoderModel.from_pretrained("Compile_model_ro")
        if type == 'initial':
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten",use_fast=True)
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print('Load Start')
    #Load the data
    test_data = getTest(csv_filename,dir,type_fn,processor)
    print('Load Completed')

    predictions = []
    actual = []
    similarities = []

    run = wandb.init(project="Trocr", name="default_model Evaluation")

    bs = 2
    for i in tqdm(range(0, len(test_data), bs)):
        batch = test_data[i:i + bs]
        pixel_batch = torch.stack([item["pixel_values"].to(torch.float32) for item in batch]).to(device)

        with torch.no_grad():
            res = model.generate(pixel_batch)
            pred = processor.tokenizer.batch_decode(res, skip_special_tokens=True)

        for j, output in enumerate(res):
            pred_res = pred[j]
            label_ids = batch[j]['labels']
            label_ids = [l for l in label_ids if l != -100]
            actual_res = processor.tokenizer.decode(label_ids, skip_special_tokens=True)
            print('pred_res:' + pred_res +'    actual_res:'+actual_res)
            predictions.append(pred_res)
            actual.append(actual_res)
            similarity = calculate_similarity(pred_res.strip().lower(), actual_res.lower())
            similarities.append(similarity)
            if len(predictions) <= 50:
                run.log({
                    "id": i + j,
                    "pred": pred,
                    "actual": actual
                })

        del pixel_batch, res, batch
        torch.cuda.empty_cache()
    
    # Calculate metrics
    preds_np = np.array(predictions)
    labels_np = np.array(actual)
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
    })
    
    print(f"Exact Match Accuracy: {accuracy:.4f}")
    print(f"Average Similarity Score: {avg_similarity:.4f}")
    print(f"Percentage of High Similarity (>0.8): {high_similarity:.4f}")

    run.finish()
