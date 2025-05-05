# Fine-Tuning and Optimizing TrOCR Model focusing on handwritten-image recognition task

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
