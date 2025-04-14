import pandas as pd
import os
from PIL import Image
import numpy as np
from transformers import TrOCRProcessor


#Load the filename and the labels
def getData(fn,dir,typefn):
    path = os.path.join(dir,fn)
    image = os.path.join(dir,typefn)

    df = pd.read_csv(path)
    #Create the actual path of the filename
    df['file'] = df['FILENAME'].apply(lambda x: os.path.join(image, x))

    labels = df['IDENTITY']
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")

    def pixel_conversion(filepath):
        try:
            with Image.open(filepath) as img:
                img = img.convert('RGB') 
                img = img.resize((384, 384)) 
                return processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    df['pixels'] = df['file'].apply(pixel_conversion)
    df = df.drop(columns=['IDENTITY','FILENAME','file'])


    return df, labels



