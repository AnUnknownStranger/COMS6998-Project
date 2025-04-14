import pandas as pd
import os
from PIL import Image
import numpy as np
from transformers import TrOCRProcessor
import torch

#Load the filename and the labels
def getData(fn,dir,typefn):
    path = os.path.join(dir,fn)
    image = os.path.join(dir,typefn)

    img_path = "pixels_df.pt"
    label_path = "labels.pt"

    if os.path.exists(img_path) and os.path.exists(label_path):
        return torch.load(img_path), torch.load(label_path)
    
    df = pd.read_csv(path)
    #Create the actual path of the filename
    df['file'] = df['FILENAME'].apply(lambda x: os.path.join(image, x))

    labels = df['IDENTITY']
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

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

    torch.save(df, img_path)
    torch.save(labels, label_path)
    return df, labels

