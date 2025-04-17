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
    type_fn = "test_v2/test" 
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
    run = wandb.init(project="Trocr", name="default_model Evaluation")

    bs = 2
    for i in tqdm(range(0,len(test_data),bs)):
        #Make a stack of batch
        batch = test_data[i:i+bs]
        pixel_batch = torch.stack([item["pixel_values"] for item in batch]).to(device)

        with torch.no_grad():
            res = model.generate(pixel_batch)
            pred = processor.tokenizer.batch_decode(res, skip_special_tokens=True)

        for j, output in enumerate(res):
            pred_res = pred[j]
            actual_res = processor.tokenizer.decode(batch[j]['labels'], skip_special_tokens=True)
            actual.append(actual_res)
            predictions.append(pred_res)
            if len(predictions)<50:
                run.log({"sample_id": i,"actual": actual_res,"pred": pred_res})

        del pixel_batch, res, batch
        torch.cuda.empty_cache()
    
    predictions_np = np.array(predictions)
    actual_np = np.array(actual)
    exact_match = predictions_np == actual_np
    accuracy = np.mean(exact_match)

    run.log({"Test Accuracy": accuracy})
    print(f"Accuracy: {accuracy}")

    run.finish()
