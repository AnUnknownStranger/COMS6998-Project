# Fine-Tuning and Optimizing TrOCR Model focusing on handwritten-image recognition task


# Description
This project focuses on fine-tuning the TrOCR model(https://huggingface.co/microsoft/trocr-base-handwritten) for handwritten name recognition using the Kaggle handwritten name dataset(https://www.kaggle.com/datasets/landlord/handwriting-recognition). Within this project, we'll explore the effect of quantization, pruning, and torch.compile() on the model's training time, accuracy, and prediction time under the same environment. We'll be utilizing a virtual machine with a setup of 1 Nividia T4 GPU, 4 cores vCPU, and a 30GB Memory from google cloud platform to perform model fine-tuning and optimization. 


Repo Root                                                                                                                                                                                                                                                                                                                                    
│                                                                                                                                                                                                                                                                                                                                            
├── Model/                                                                                                                                                                                                                                                                                                                                   
│   ├── preprocess/     
│   │   ├── download.py    #to download the dataset                                                                                                                                                                                                                                                                                          
│   │   └── preprocess.py  #util for preprocessing the images                                                                                                                                                                                                                                                                                
│   ├── Compile.py         #applying torch.compile technique on the TrOCR model                                                                                                                                                                                                                                                              
│   ├── Default.py         #fine-tune the model without further optimization techniques                                                                                                                                                                                                                                                      
│   ├── Predict.py         #Evlaute the model on the testing set                                                                                                                                                                                                                                                                             
│   ├── Pruning.py         #applying pruning technique                                                                                                                                                                                                                                                                                       
│   └── Quantization.py    #apply quantization technique                                                                                                                                                                                                                                                                                     
│                                                                                                                                                                                                                                                                                                                                            
└── Readme.me            


# Commands to execute the code

1. Download the data by running ' python download.py ' under Model/preprocess folder
2. Copy the directory link
3. Manually modify 'dir' 'csv_filename' and 'type_fn' variables under each python within within Model folder
4. Run ' python Default.py ' fine-tune the model on the datasets

Compile.py
  - Run ' python Compile.py default ' to fine-tune on the default compiled model
  - Run ' python Compile.py ro' to fine-tune on the reduce-overhead mode
  - Run ' python Compile.py ma' to fine-tune on the max-autotune mode

Prediction.py
  - Run ' python Predict.py ' to perform evaluation on the default fine-tuned model
  - Run ' python Predict.py default ' to perform evaluation on the default compiled model
  - Run ' python Predict.py ro' to perform evaluation on fine tuned compiled model in reduce-overhead mode
  - Run ' python Predict.py ma' to perform evaluation on fine tuned compiled model in max-autotune mode
  - Run ' python Predict.py Fdefault' to perform evaluation on the default compilation wrapper on the fine-tuned model
  - Run ' python Predict.py Fro' to perform evaluation on max-autotune compilation wrapper on fine-tuned model
  - Run ' python Predict.py Fma' to perform evaluation on the reduce-overhead compilation wrapper on the fine-tuned model

Quantization.py
  - Run ' python Quantization.py ' to optimize the model with dynamic quantization


# Result
![image](https://github.com/user-attachments/assets/8ae04361-f66f-442a-8e61-aa724c3f15d7)

![image](https://github.com/user-attachments/assets/205a411a-eb34-4dfb-9091-caabf46276ac)






# Wave and Biases Link
Prediction workspace(https://wandb.ai/wei1070580217-columbia-university/Trocr?nw=nwuserwei1070580217)
Training workspace(https://wandb.ai/wei1070580217-columbia-university/huggingface?nw=nwuserwei1070580217)
