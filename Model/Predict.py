import torch
from transformers import VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator,TrOCRProcessor
import wandb
import os
from preprocess.preprocess import getTest
import time
from tqdm import tqdm
import numpy as np

if __name__ == "__main__":
    torch.cuda.empty_cache()
    dir = "/home/wei1070580217/.cache/kagglehub/datasets/landlord/handwriting-recognition/versions/1"

    csv_filename = "written_name_test_v2.csv"
    type_fn = "test_v2/train" 
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    model = VisionEncoderDecoderModel.from_pretrained("default_model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    print('Load Start')
    #Load the data
    test_data = getTest(csv_filename,dir,type_fn,processor)
    print('Load Completed')

    predictions = []
    actual = []

    for i, item in tqdm(enumerate(test_data), total=len(test_data)):
        pixel_values = item["pixel_values"].to(device)
        label_ids = item["labels"].to(device)

        with torch.no_grad():
            res = model.generate(pixel_values)
        
        pred_res = processor.tokenizer.batch_decode(res, skip_special_tokens=True)[0]
        actual_res = processor.tokenizer.decode(label_ids[0], skip_special_tokens=True)

        actual.append(actual_res)
        predictions.append(pred_res)

        wandb.log({"sample_id": i,"actual": actual_res,"pred": pred_res})
    
    predictions_np = np.array(predictions)
    actual_np = np.array(actual)
    exact_match = predictions_np == actual_np
    accuracy = np.mean(exact_match)

    wandb.log({"Test Accuracy": accuracy})
    print(f"Accuracy: {accuracy}")

    wandb.finish()
