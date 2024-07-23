## Software Requirement
```
numpy== 1.25.0
torch==1.13.0+cu116
python==3.9.18
sklearn==0.22.0
pandas==2.1.3
cuda=11.6
pyod==1.0.9
adbench==0.1.11
deepod==0.4.1
```
## Get Start

Download data and put them into the folder: datasets/.

Download [well-trained-models](https://drive.google.com/drive/folders/183dC-db9C6S7iJkOrTrVrPTVTyikBztb?usp=drive_link) and put them into the folder: Well_Trained_Models/.

## Train the model

To reproduce our results, run Main.py and set compare_alg = 0.

To compare the performance of our method, run Main.py and set compare_alg = 1.

