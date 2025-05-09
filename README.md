# HPML Project: [Fine-Tuning and Optimizing TrOCR Model focusing on handwritten-image recognition task]
- **Team Name**: [HPML]
- **Members**:
  - Wei Chen (wc2917)
  - Tao Tong (tt3310)
  - Haotian Zhang (hz2294)

---

## 1. Problem Statement
In modern era, even through the rapid development of technology, handwriting still remains a common method of communication especially in education. Handwritten identity/name recognition remains a part of the crucial aspect of academic record management. However, the manual transcription of written words is time consuming. Therefore, developing an efficient machine learning model can significantly improve the general process.

This project focuses on fine-tuning the TrOCR model(https://huggingface.co/microsoft/trocr-base-handwritten) for handwritten name recognition using the Kaggle handwritten name dataset(https://www.kaggle.com/datasets/landlord/handwriting-recognition). Within this project, we'll explore the effect of quantization, pruning, and torch.compile() on the model's training time, accuracy, and prediction time under the same environment. We'll be utilizing a virtual machine with a setup of 1 Nividia T4 GPU, 4 cores vCPU, and a 30GB Memory from google cloud platform to perform model fine-tuning and optimization. 

---

## 2. Model Description
Microsoft TrOCR Model(https://huggingface.co/microsoft/trocr-base-handwritten) 
- Framework: Pytorch
---

## 3. Final Results Summary

Microsoft TrOCR Model
| Metric                                         | Value                                    |
|------------------------------------------------|------------------------------------------|
| Average Similarity Score(Levenshtein Distance) | 45.581%                                  |
| Testing   Time                                 | 29m 58s                                  |
| Training Time/Epoch                            | N/A                                      |
| Device                                         | GCP, Nividia T4, 4 core CPU, 30GB Memory |

Fine-Tuned TrOCR Model
| Metric                                         | Value                                    |
|------------------------------------------------|------------------------------------------|
| Average Similarity Score(Levenshtein Distance) | 58.221%                                  |
| Testing   Time                                 | 45m 35s                                  |
| Training Time/Epoch                            | 1h 44m 48s                               |
| Device                                         | GCP, Nividia T4, 4 core CPU, 30GB Memory |

Torch.compile() default parameter TrOCR Model
| Metric                                         | Value                                    |
|------------------------------------------------|------------------------------------------|
| Average Similarity Score(Levenshtein Distance) | 57.651%                                  |
| Testing   Time                                 | 45m 44s                                  |
| Training Time/Epoch                            | 1h 36m 36s                               |
| Device                                         | GCP, Nividia T4, 4 core CPU, 30GB Memory |

Pruned TrOCR Model
| Metric                                         | Value                                    |
|------------------------------------------------|------------------------------------------|
| Average Similarity Score(Levenshtein Distance) | 46.681%                                  |
| Testing   Time                                 | 30m 05s                                  |
| Training Time/Epoch                            | 1h 44m 1s                                |
| Device                                         | GCP, Nividia T4, 4 core CPU, 30GB Memory |

Quantization Version of TrOCR Model
| Metric                                         | Value                                    |
|------------------------------------------------|------------------------------------------|
| Average Similarity Score(Levenshtein Distance) | 54.621%                                  |
| Testing   Time(on CPU)                         | 5h 17m 52s                               |
| Training Time/Epoch                            | N/A                                      |
| Device                                         | GCP, Nividia T4, 4 core CPU, 30GB Memory |

---
### A. Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

B. Wandb Dashboard

View training and evaluation metrics here: Wandb Dashboard Link
Prediction workspace(https://wandb.ai/wei1070580217-columbia-university/Trocr?nw=nwuserwei1070580217)
Training workspace(https://wandb.ai/wei1070580217-columbia-university/huggingface?nw=nwuserwei1070580217)

---
### C. Specify for Training

To train the model from scratch:
1. Download the data by running
   ```bash
   python download.py
   ```
   under Model/preprocess folder
2. Copy the directory link after the code finish running
3. Manually modify 'dir' 'csv_filename' and 'type_fn' variables under each Python file within Model folder
4. Run
   ```bash
    python Default.py
   ```
   fine-tune the model on the datasets
Compile.py
  - Run
    ```bash
    python Compile.py default
    ```
    to fine-tune on the default compiled model
    
  - Run
    ```bash
    python Compile.py ro
    ```
    to fine-tune on the reduce-overhead mode
  - Run
    ```bash
    python Compile.py ma
    ```
    to fine-tune on the max-autotune mode
Quantization.py
  - Run 
    ```bash
    python Quantization.py
    ```
    to optimize the model with dynamic quantization
Pruning.py
  - Run
    ```bash
    python Pruning.py
    ```
    to run training on pruned model


---


### D. Evaluation

To evaluate the trained model:

Prediction.py
  - Run
    ```bash
    python Predict.py
    ```
    to perform evaluation on the default fine-tuned model
  - Run
    ```bash
    python Predict.py default
    ```
    to perform evaluation on the default compiled model
  - Run
    ```bash
    python Predict.py ro
    ```
    to perform evaluation on fine tuned compiled model in reduce-overhead mode
  - Run
    ```bash
    python Predict.py ma
    ```
    to perform evaluation on fine tuned compiled model in max-autotune mode
  - Run
    ```bash
    python Predict.py Fdefault
    ```
    to perform evaluation on the default compilation wrapper on the fine-tuned model
  - Run
    ```bash
    python Predict.py Fro
    ```
    to perform evaluation on max-autotune compilation wrapper on fine-tuned model
  - Run
    ```bash
    python Predict.py Fma
    ```
    to perform evaluation on the reduce-overhead compilation wrapper on the fine-tuned model
  - Run
    ```bash
    python Predict.py pruned
    ```
    to perform evaluation on the pruned model

---
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



