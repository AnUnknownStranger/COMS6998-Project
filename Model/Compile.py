import torch
from transformers import VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator,TrOCRProcessor
import wandb
import os
from preprocess.preprocess import getData,getTest
import time
import sys
import numpy as np
from tqdm import tqdm
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
    type = str(sys.argv[1])

    torch.cuda.empty_cache()
    dir = "/home/wei1070580217/.cache/kagglehub/datasets/landlord/handwriting-recognition/versions/1"

    csv_filename = "written_name_train_v2.csv"
    type_fn = "train_v2/train" 
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten",use_fast=True)
    print('Load Start')
    #Load the data
    train_data = getData(csv_filename,dir,type_fn,processor)
    print('Load Completed')

    #Load the model from hugging face
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    #Configure the start token
    model.config.decoder_start_token_id = processor.tokenizer.pad_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    #Send the model to CPU
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    #Check the type of compiliation to execute
    if type == 'default':
        model = torch.compile(model, backend="inductor")
    if type == 'ma':
        model  = torch.compile(model, backend="inductor",mode="max-autotune")
    if type == 'ro':
        model = torch.compile(model, backend="inductor",mode="reduce-overhead")
    #Setup the training parameter
    training_args = Seq2SeqTrainingArguments(
        output_dir="./model",
        per_device_train_batch_size=4,
        eval_steps=500,
        num_train_epochs=10,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        learning_rate=1e-5,
        report_to="wandb",
        logging_steps=50,
        save_safetensors=False
        )
    
    #setup the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=processor.tokenizer,
        data_collator=default_data_collator,
    )
    #Calculate the training time and execute the training
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Training completed in {time_taken} seconds")
    
    #Save the fine-tuned model
    trainer.model.save_pretrained(f"Compile_model_{type}")
    processor.save_pretrained(f"Compile_model_{type}")  

    

    



    
