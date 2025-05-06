# Fine-Tuning and Optimizing TrOCR Model focusing on handwritten-image recognition task


# Description
This project focuses on fine-tuning the TrOCR model(https://huggingface.co/microsoft/trocr-base-handwritten) for handwritten name recognition using Kaggle handwritten name dataset(https://www.kaggle.com/datasets/landlord/handwriting-recognition). Within this project, we'll explore the effect of quantization, pruning, and torch.compile() on the model's training time, accuracy, and prediction time under the same environment. We'll be utilizing virtual machine with a setup of Nividia T4 GPU, 4 cores vCPU, and a 30GB Memory from google cloud platform to perform model fine-tunuing and optimization. 


# Repo Outline
|-Model
  |-preprocess
    |-download.py
    |-preprocess.py
  |-Compile.py
  |-Default.py
  |-Predict.py
  |-Pruning.py
  |-Quantization.py
|-Readme.me

# Commands to execute the code

1. Download the data by running ' python download.py ' under Model/preprocess folder
2. Copy the directory link
3. Manually modify 'dir' 'csv_filename' and 'type_fn' variables under each python within within Model folder
4. Run ' python Default.py ' fine-tune the model on the datasets

Compile.py
  - Run ' python Compile.py default ' to fine-tune on the default compiled model
  - Run ' python Compile.py ro' to fine-tune on the on the reduce-overhead mode
  - Run ' python Compile.py ma' to fine-tune on the max-autotune mode

Prediction.py
  - Run ' python Predict.py ' to perform evaluation on default fine-tuned model
  - Run ' python Predict.py default ' to perform evaluation on default compiled model
  - Run ' python Predict.py ro' to perform evaluation on fine tuned compiled model on reduce-overhead mode
  - Run ' python Predict.py ma' to perform evaluation on fine tuned compiled model on max-autotune mode


# Result
------------------Model--------------|----Training Time----|----Prediction Time----|----Average Accuracy----|
Microsoft TrOCR handwritten Ver.     |         NA          |         28m 58s       |         0.45581        |
Fine-Tuned TrOCR Model               |      1h 44m 48s     |         45m 35s       |         0.58221        |
Compiled Ver.                        |      1h 36m 36s     |         45m 44s       |         0.57651        |


# Wave and Biases Link
Prediction workspace(https://wandb.ai/wei1070580217-columbia-university/Trocr?nw=nwuserwei1070580217)
Training workspace(https://wandb.ai/wei1070580217-columbia-university/huggingface?nw=nwuserwei1070580217)