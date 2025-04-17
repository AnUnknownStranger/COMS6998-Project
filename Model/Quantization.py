import torch
from transformers import VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator, TrOCRProcessor
import os
from preprocess.preprocess import getData
import time
import wandb
from torch.quantization import quantize_dynamic
from torch.quantization import prepare_qat, convert
import traceback
import json

def quantize_model(model):
    """
    Perform dynamic quantization on the model
    """
    print("Starting model quantization...")
    try:
        quantized_model = quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        print("Model quantization completed successfully")
        return quantized_model
    except Exception as e:
        print(f"Error during model quantization: {str(e)}")
        print(traceback.format_exc())
        raise e

def prepare_model_for_quantization(model):
    """
    Prepare model for quantization-aware training
    """
    print("Preparing model for quantization...")
    try:
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        model_prepared = prepare_qat(model)
        print("Model preparation completed successfully")
        return model_prepared
    except Exception as e:
        print(f"Error during model preparation: {str(e)}")
        print(traceback.format_exc())
        raise e

def save_quantized_model(model, save_path):
    """
    Save quantized model using torch.save
    """
    print(f"Saving quantized model to {save_path}...")
    os.makedirs(save_path, exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
    
    # Save config
    config = model.config.to_dict()
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(config, f)
    
    print("Model saved successfully")

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="handwriting-recognition", name="quantization_training")
    
    torch.cuda.empty_cache()
    dir = "/home/tt3010/COMS6998-Project/dataset"
    
    try:
        # Load data
        print('Loading data...')
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten", use_fast=True)
        train_data = getData(
            "written_name_train_v2.csv",
            dir,
            "train_v2/train", 
            processor
        )
        print('Data loading completed')
        
        # Load and prepare model
        print('Loading model...')
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        model.config.decoder_start_token_id = processor.tokenizer.pad_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        
        print('Preparing model for quantization...')
        model = prepare_model_for_quantization(model)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training configuration
        training_args = Seq2SeqTrainingArguments(
            output_dir="./quantized_model",
            per_device_train_batch_size=4,
            eval_steps=500,
            num_train_epochs=5,
            save_steps=500,
            save_total_limit=2,
            logging_dir="./quantized_logs",
            remove_unused_columns=False,
            fp16=torch.cuda.is_available(),
            report_to="wandb",
            logging_steps=50,
            run_name="quantization_training",
            save_safetensors=False  # Disable safe tensor serialization
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            tokenizer=processor.tokenizer,
            data_collator=default_data_collator,
        )
        
        # Start training
        print('Starting quantization-aware training...')
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Quantization-aware training completed in {time_taken} seconds")
        
        # Convert and save quantized model
        print('Converting model to quantized model...')
        try:
            quantized_model = convert(model)
            print("Model conversion completed successfully")
            
            # Save quantized model using custom function
            save_quantized_model(quantized_model, "./quantized_model/final")
            
            # Compare model sizes
            print("\nCalculating model sizes...")
            original_size = sum(p.numel() * p.element_size() for p in model.parameters())
            quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
            print("\n" + "="*50)
            print("Model Size Comparison:")
            print(f"Original model size: {original_size / 1024 / 1024:.2f} MB")
            print(f"Quantized model size: {quantized_size / 1024 / 1024:.2f} MB")
            print(f"Compression ratio: {original_size / quantized_size:.2f}x")
            print("="*50 + "\n")
            
        except Exception as e:
            print(f"Error during model conversion or size comparison: {str(e)}")
            print(traceback.format_exc())
            raise e
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())
        raise e
    finally:
        wandb.finish() 