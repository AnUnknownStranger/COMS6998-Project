import pandas as pd
import os
from PIL import Image
import numpy as np
from transformers import TrOCRProcessor
import torch

#Load the filename and the labels
def getData(fn,dir,typefn,processor):
    path = os.path.join(dir,fn)
    image = os.path.join(dir,typefn)

    img_path = "train_data.pt"

    if os.path.exists(img_path):
        return torch.load(img_path)
    
    df = pd.read_csv(path)
    #Create the actual path of the filename
    df['file'] = df['FILENAME'].apply(lambda x: os.path.join(image, x))

    labels = df['IDENTITY']

    def pixel_conversion(filepath):
        try:
            with Image.open(filepath) as img:
                img = img.convert('RGB') 
                return processor(images=img, return_tensors="pt").pixel_values
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    df['pixels'] = df['file'].apply(pixel_conversion)
    #Delete all the null column
    df = df[df['pixels'].notnull()].reset_index(drop=True)
    df = df.drop(columns=['IDENTITY','FILENAME','file'])

    #Create the Train_data
    train_data = []
    for img, label in zip(df['pixels'], labels):
        encoding = processor.tokenizer(label,padding="max_length",max_length=128,truncation=True,return_tensors="pt")
        train_data.append({"pixel": image,"labels": encoding.input_ids.squeeze(),"attention_mask": encoding.attention_mask.squeeze()})

    torch.save(img_path)
    return train_data

