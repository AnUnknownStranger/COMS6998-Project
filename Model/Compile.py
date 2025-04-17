import torch
from transformers import VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator,TrOCRProcessor
import wandb
import os
from preprocess.preprocess import getData
import time


if __name__ == "__main__":
    torch.cuda.empty_cache()
    dir = "/home/wei1070580217/.cache/kagglehub/datasets/landlord/handwriting-recognition/versions/1"

    csv_filename = "written_name_train_v2.csv"
    type_fn = "train_v2/train" 
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten",use_fast=True)
    print('Load Start')
    #Load the data
    train_data = getData(csv_filename,dir,type_fn,processor)
    print('Load Completed')
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    model.config.decoder_start_token_id = processor.tokenizer.pad_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    model = torch.compile(model, backend="inductor")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./model",
        per_device_train_batch_size=4,
        eval_steps=500,
        num_train_epochs=5,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        report_to="wandb",
        logging_steps=50,
        save_safetensors=False
        )
    
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        tokenizer=processor.feature_extractor,
        data_collator=default_data_collator,
    )
    start_time = time.time()
    trainer.train()
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Training completed in {time_taken} seconds")

    trainer.save_model("Compile_model_default")
    



    
