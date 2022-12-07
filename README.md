# smai-project-m-22

## Similarity-Aware Deep Attentive Model for Clickbait Detection
### Team Name: Ego et al.
<b> Team Members: - </b><br>
Tanmay Bhatt <br>
Anjali Singh <br>
Pranav Manu <br>
Abhinav Siddharth <br> 

## Clone the Repo
`git clone https://github.com/anjalish05/smai-project-m-22 --depth 1`

## Summary of the Repo
- `src` contains all the code files
- `Papers` contains the Project Research Paper
- `Documents` contains the ppts and report for the project
- `Outputs` contains the plots and results

## Summary of `src` folder
- `input_functions.py` contains the functions used for preprocessing and converting the data into vector form (for the Clickbait 17 set)
- `model.py` contains the functions for defining the LSD and LSDA models
- `traditional_models.py` contains the functions for testing the traditional models like SVM, Logistic Regression and Random Forest
- `clickbait_training.ipynb` is the notebook which contains the code for training the model on the data
- `train.py` is a python script version of `clickbai_training.ipynb` which will train and save the model as a checkpoint
- `NetModel.pth`, `NetModel2.pth`, `checkpoint.pth` and `checkpoint_nlayer_2.pth` are trained model checkpoints
- `clickbait_testing.ipynb` is the notebook which contains teh code for testing the model on the data and predicting the labels
- `metrics.py` contains functions for testing the validity of the model
- `FNC_Processing.ipynb` is the notebook which contains the code for processing and obtaining word vectors for the FNC dataset. Once the word vectors are extracted, the method for training and testing is identical as in the case of the Clickbait-17 set

## Links for the Datasets
- Clickbait Challenge 2017 - https://zenodo.org/record/5530410#.Y3JTGcdBxPZ
- Fake News Challenge 2017 (FNC) - https://github.com/FakeNewsChallenge/fnc-1
