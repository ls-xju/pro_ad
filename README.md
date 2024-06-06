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

Download [data](https://github.com/Minqi824/ADBench/tree/main/adbench/datasets/Classical) and put them into the folder: datasets/.

## Train the model

To reproduce our results, run Running_Load.py.

To compare the performance of our method and baseline methods, run Running_Main.py and set compare_alg=0 for our method and compare_alg=1 for baseline methods, respectively.

