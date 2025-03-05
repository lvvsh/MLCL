# MLCL

## The Frame of MLCL
![](https://github.com/lvvsh/MLCL/blob/main/Frame.png)

# Usage
## Requirements

1.Create a conda environment with python 3.11.

```python
conda create -n MLCL python=3.11
```

2.Activate the environment

```python
conda activate MLCL
````

3.Install the dependencies

```python
pip install -r requirements.txt
```

4.Clone the repository and change the directory

```python
https://github.com/lvvsh/MLCL.git

cd MLCL
```
## Training

### Dataset Preparation

1.LEVIR:https://github.com/justchenhao/LEVIR

2.WHU:http://gpcv.whu.edu.cn/data/building_dataset.html

3.SYSU:https://github.com/liumency/SYSU-CD

### train

1.train
```python
python main_train.py
```
## Experiment
### result on the LEVIR-CD
| Method | Pre. | Rec. | F1 | IoU | OA |
|--------|------|------|-----|-----|-----|
| FC-EF | 84.18 | 79.89 | 81.98 | 69.46 | 98.21 |
| IFNet | 92.00 | 83.58 | 87.58 | 77.91 | 98.80 |
| SNUNet | 90.71 | 88.75 | 89.72 | 81.36 | 98.96 |
| BIT | 91.39 | 88.30 | 89.82 | 81.52 | 98.98 |
| ChangeFormer | 90.63 | 87.92 | 89.26 | 80.60 | 98.92 |
| DMINet | 90.85 | 88.96 | 89.90 | 81.64 | 98.98 |
| TFI-GR | 92.49 | 88.94 | 90.68 | 82.95 | **98.99** |
| A2Net | 92.96 | 85.81 | 89.24 | 80.58 | 98.98 |
| SEIFNet | 92.49 | 89.46 | 90.95 | 83.40 | 98.09 |
| Ours | **95.57** | **92.24** | **93.83** | **88.89** | 98.85 |

### result on the SUSU-CD
| Method       | Pre.  | Rec.  | F1    | IoU   | OA    |
|--------------|-------|-------|-------|-------|-------|
| FC-EF   | 83.91 | 66.98 | 74.49 | 59.63 | 89.18 |
| IFNet      | 82.78 | 69.55 | 75.59 | 60.76 | 89.41 |
| SNUNet     | 80.04 | 79.27 | 79.65 | 66.19 | 90.45 |
| BIT       | 79.73 | 74.79 | 77.18 | 62.84 | 89.57 |
| ChangeFoemer | 80.47 | 74.23 | 77.23 | 62.90 | 89.68 |
| DMINet       | 84.19 | 78.10 | 81.19 | 68.11 | 91.37 |
| TFI-GR     | 86.09 | 76.14 | 80.81 | 67.79 | 91.47 |
| A2Net       | 84.69 | 78.75 | 81.61 | 68.93 | 91.63 |
| SEIFNet      | 84.81 | 79.98 | 82.32 | 69.96 | **91.90** |
| Ours         | **87.69** | **87.34** | **87.51** | **78.41** | 91.03 |

### result on the WHU-CD
| Method       | Pre.  | Rec.  | F1    | IoU   | OA    |
|--------------|-------|-------|-------|-------|-------|
| FC-EF   | 67.09 | 68.15 | 67.61 | 51.07 | 97.35 |
| IFNet   | 95.72 | 75.97 | 84.71 | 73.47 | 98.88 |
| SNUNet| 84.62 | 82.64 | 83.52 | 71.71 | 98.68 |
| BIT     | 88.07 | 81.47 | 84.64 | 73.38 | 98.80 |
| ChangeFoemer | 84.37 | 77.33 | 80.69 | 67.64 | 98.50 |
| DMINet  | 85.12 | 80.43 | 82.70 | 70.51 | 98.64 |
| TFI-GR  | 77.40 | 87.69 | 82.22 | 69.81 | 98.46 |
| A2Net | 90.70 | 81.03 | 85.60 | 74.81 | 98.89 |
| SEIFNet | 87.01 | 85.77 | 86.39 | 76.04 | 98.90 |
| ours         | **96.10** | **94.15** | **95.10** | **91.00** | **99.27** |


