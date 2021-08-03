# [Real-world Sentence Boundary DetectionUsing Multitask Learning: A Case Study on French]
This repository is for "Real-world Sentence Boundary DetectionUsing Multitask Learning: A Case Study on French" (submitted to NLE, CUP). You can check the data in the "data" folder both for training and testing. 
We serve around 50 percent of our training data because of license issues. You can test three different models, namely 'Baseline', 'multi-task with XLM-Roberta', and 'multi-task with CamemBERT' models.


## 1. Dependencies

This is a list of packages that required to run the codes.

 * Python 3.7 interpreter
 * Pytorch 1.6.0
 * transformers from Huggingface
 

### 1-1. Install Pytorch
 * Windows, Linux
  ```
   conda install pytorch-cpu torchvision-cpu -c pytorch
  ```
 * MacOS
  ```
   conda install pytorch torchvision -c pytorch
  ```
  

### 1-2. Install Required packages
 * Windows, Linux
  ```
   pip install -r requirements.txt
  ```

## 2. Training and testing

### 2-1. Baseline with XLM-Roberta
  ```
   python baseline.py
  ```
### 2-2. Multi-task with XLM-Roberta
  ```
   python multi-task-roberta.py
  ```
### 2-3. Multi-task with CamemBERT
  ```
   python multi-task-camembert.py
  ```
  
## 3. Performance
![initial](https://user-images.githubusercontent.com/4470398/99162632-2f9c5a00-2743-11eb-845e-b0045f8be002.png)
