import torch
from transformers import VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator,TrOCRProcessor
import wandb
import os
from preprocess.preprocess import getData
import time


if __name__ == "__main__":
    #Empty the cache
    torch.cuda.empty_cache()
    #Setup directory
    dir = "/home/wei1070580217/.cache/kagglehub/datasets/landlord/handwriting-recognition/versions/1"
    csv_filename = "written_name_train_v2.csv"
    type_fn = "train_v2/train" 
    #Load the processor from hugging face
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten",use_fast=True)
    print('Load Start')
    #Load the data
    train_data = getData(csv_filename,dir,type_fn,processor)
    print('Load Completed')
    #Load the TrOCR model from hugging face
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    #Configure the start token
    model.config.decoder_start_token_id = processor.tokenizer.pad_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    #Send the model to GPU
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    #Configure the hyperparameter
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
    #Configure the trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=processor.feature_extractor,
        data_collator=default_data_collator,
    )
    #Calculate the training time
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Training completed in {time_taken} seconds")
    #Save the model
    trainer.save_model("default_model")
    



    
