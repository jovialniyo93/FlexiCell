![Python](https://img.shields.io/badge/python-v3.11-blue)
![Pytorch](https://img.shields.io/badge/Pytorch-V1.6-orange)
![Opencv-python](https://img.shields.io/badge/OpenCV-V4.8.0.76-brightgreen)
![pandas](https://img.shields.io/badge/Pandas-V1.4.2-ff69b4)
![numpy](https://img.shields.io/badge/%E2%80%8ENumpy-V1.20.2-success)
![releasedate](https://img.shields.io/badge/release%20date-October%2024-red)
![Opensource](https://img.shields.io/badge/OpenSource-Yes!-6f42c1)

# FlexiCell: Deep Learning with Learnable Adaptive Filtering and Dual Attention for Cell Segmentation #

The code in this repository is supplementary to our future publication ***FlexiCell: Deep Learning with Learnable Adaptive Filtering and Dual Attention for Cell Segmentation*** 

  <div align="center">
    <img src="visualize.gif", width="600">
  </div>


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

## Prerequisites
* [Anaconda Distribution](https://www.anaconda.com/products/individual)
* A CUDA capable GPU
* Minimum / recommended RAM: 16 GiB / 32 GiB
* Minimum / recommended VRAM: 12 GiB / 24 GiB
* This project is writen in Python 3 and makes use of Pytorch. 

## Installation
In order to get the code, either clone the project, or download a zip file from GitHub.

Clone the transdeepseg repository:
```
https://github.com/jovialniyo93/FlexiCell.git
```
Open the Anaconda Prompt (Windows) or the Terminal (Linux), go to the FlexiCell repository and create a new virtual environment:
```
cd path_to_the_cloned_repository
```
```
conda env create -f requirements.yml
```
Activate the virtual environment FlexiCell_ve:
```
conda activate FlexiCell_ve
```

# How to train and test our model

1. Data augmentation

python augmentation.py

2. Model training

python train.py

3. Model Testing

python test.py

4. Evaluate using Cell-Tracking-Challenge, the MOTChallenge, and the CHOTA metric.
The package enables the validation, evaluation, and visualization of tracking results. Below are examples provided for a sample directory organized in the CTC format as follows:
```bash
ctc
├── train
│   ├── challenge_x
│   │   ├── 01
│   │   ├── 01_GT
│   │   ├── 01_RES
│   │   
│   ├── challenge_y 
│   │   ├── 01
│   │   ├── 01_GT
│   │   ├── 01_RES
```
The directory ```ctc```  contains the training data. The subdirectories 
```challenge_x``` and ```challenge_y``` contain the data for the different
challenges. Each challenge directory contains subdirectories for the sequences
```01_GT```, ```01_RES```. The directories 
```01_GT``` and ```01_RES``` contain the ground truth and the tracking results
for the sequence ```01```. 
pip install py-ctcmetrics

i) To validate if challenge_x/01 is correctly formatted, run the command

ctc_validate --res "ctc/train/challenge_x/01_RES"

ii) Moreover, you can recursively validate the tracking results for all challenges/sequences in a directory by adding the flag -r:
ctc_validate --res "ctc/train" -r

iii) To evaluate results against the ground truth, similar commands can be used. For example, to evaluate the sequence challenge_x/01, run the command

ctc_evaluate --gt "ctc/train/challenge_x/01_GT" --res "ctc/train/challenge_x/01_RES"

iv) or recursively for all sequences in a directory:
ctc_evaluate --gt "ctc/train" --res "ctc/train" -r

v)Per default, the code is executed using multiple processes with one process per available CPU core. Multiprocessing decreases the execution time but also increases the memory consumption.
ctc_evaluate --gt "ctc/train" --res "ctc/train" -r -n 4

vi)The evaluation results are printed to the console. If you want to save the results to a csv file, you can use the argument --csv-file:

ctc_evaluate --gt "ctc/train" --res "ctc/train" -r --csv-file "ctc/results.csv"

**Note:** The csv file will be overwritten if it already exists!
The following table shows the available arguments:

| Argument      | Description                                          | Default |
|---------------|------------------------------------------------------| --- |
| --gt          | Path to the ground truth directory.                  | None |
| --res         | Path to the results directory.                       | None |
| --recursive   | Recursively evaluate all sequences in the directory. | False |
| --csv-file    | Path to a csv file to save the results.              | None |
| --num-threads | Number of threads to use for evaluation.             | 1    |

By default, all provided metrics are evaluated. However, you can choose specific metrics to focus on, 
allowing you to skip the calculation of metrics that are not relevant to your interests. 
The additional arguments for selecting a specific subset of metrics are:

| Argument | Description                                                     | 
| --- |-----------------------------------------------------------------|
| --valid | Check if the result has valid format                            | 
| --det | The DET detection metric                                        |
| --seg | The SEG segmentation metric                                     |
| --tra | The TRA tracking metric                                         |
| --lnk | The LNK linking metric                                          |
| --ct | The CT (complete tracks) metric                                 |
| --tf | The TF (track fraction) metric                                  |
| --bc | The BC(i) (branching correctness) metric                        |
| --cca | The CCA (cell cycle accuracy) metric                            |
| --mota | The MOTA (Multiple Object Tracking Accuracy) metric             |
| --hota | The HOTA (Higher Order Tracking Accuracy) metric                |
| --idf1 | The IDF1 (ID F1) metric                                         |
| --chota | The CHOTA (Cell-specific Higher Order Tracking Accuracy) metric |
| --mtml | The MT (Mostly Tracked) and ML (Mostly Lost) metrics            |
| --faf | The FAF (False Alarm per Frame) metric                          |
---

vii) To properly evaluate the SEG (segmentation) metric using the ctc_evaluate script, you need to ensure that you explicitly pass the --seg flag during the evaluation. 

ctc_evaluate --gt "ctc/train/challenge_x/01_GT" --res "ctc/train/challenge_x/01_RES" --seg

viii) To evaluate the SEG (segmentation) metric using the command you provided, you need to add the --seg flag, which is specific for segmentation evaluation.  command for segmentation evaluation:

ctc_evaluate --gt "ctc/train" --res "ctc/train" -r --csv-file "ctc/results.csv" --seg

ix) Command for SEG-Only Evaluation:
ctc_evaluate --gt "ctc/train/challenge_x/01_GT" --res "ctc/train/challenge_x/01_RES" --seg --csv-file "ctc/results_seg.csv"
ctc_evaluate --gt "ctc/train/challenge_x/01_GT" --res "ctc/train/challenge_x/01_RES" --seg
ctc_evaluate --gt "ctc/train/challenge_x/01_GT" --res "ctc/train/challenge_x/01_RES" --seg --csv-file "ctc/results_seg.csv"

x)You can visualize your tracking results with the following command:
ctc_visualize --img "ctc/train/challenge_x/01" --res "ctc/train/challenge_x/01_RES"

xi)To save the visualized image results to a specific folder using the ctc_visualize command, you can use the --viz. Here’s how to modify your command:
ctc_visualize --img "ctc/train/challenge_x/01" --res "ctc/train/challenge_x/01_RES" --viz "visualize/01"

xii)Great! Now that you’re able to save the visualized images, to save a video of the visualization as well, you just need to include the --video-name flag in your command. 

ctc_visualize --img "ctc/train/challenge_x/01" --res "ctc/train/challenge_x/01_RES" --viz "visualize/01" --video-name "visualize/01/visualization.mp4"
The command will show the visualizations of the tracking results. You can 
control the visualization with specific keys:


| Key   | Description                                                         | 
|-------|---------------------------------------------------------------------| 
| q     | Quits the Application                                               |  
| w     | Start or Pause the auto visualization                               |
| d     | Move to the next frame                                              |
| a     | Move to the previous frame                                          |
| l     | Toggle the show labels option                                       |
| p     | Toggle the show parents option                                      |
| s     | Save the current frame to the visualization directory as .jpg image |


Additional arguments can be used to customize the visualization. 
The table below lists the available options:


| Argument          | Description                                                                              | Default |
|-------------------|------------------------------------------------------------------------------------------|---------|
| --img             | The directory to the images **(required)**                                               |         |
| --res             | The directory to the result masks **(required)**                                         |         |
| --viz             | The directory to save the visualization                                                 | None    |
| --video-name      | The path to the video if a video should be created                                       | None    |
| --border-width    | The width of the border. Either an integer or a string that describes the challenge name | None    |
| --show-no-labels  | Print no instance labels to the output as default                                        | False   |
| --show-no-parents | Print no parent labels to the output as default                                          | False   |
| --ids-to-show     | The IDs of the instances to show. If defined, all others will be ignored.                | None    |
| --start-frame     | The frame to start the visualization                                                     | 0       |
| --framerate       | The framerate of the video                                                               | 10      |
| --opacity         | The opacity of the instance colors                                                       | 0.5     |

To use the evaluation protocol in your python code, the code can be imported
as follows:

```python   
from ctc_metrics import evaluate_sequence, validate_sequence

# Validate the sequence
res = validate_sequence("/ctc/train/challenge_x/01_RES")
print(res["Valid"])

# Evaluate the sequence
res = evaluate_sequence("/ctc/train/challenge_x/01_GT", "/ctc/train/challenge_x/01_RES")
print(res["DET"])
print(res["SEG"])
print(res["TRA"])
...
    
```
**Created by:** Ph.D. student: Jovial Niyogisubizo 
Department of Computer Applied Technology,  
Center for High Performance Computing, Shenzhen Institute of Advanced Technology, CAS. 

For more information, contact:

* **Prof Yanjie Wei**  
Shenzhen Institute of Advanced Technology, CAS 
Address: 1068 Xueyuan Blvd., Shenzhen, Guangdong, China
Postcode: 518055
yj.wei@siat.ac.cn


* **Jovial Niyogisubizo**  
Shenzhen Institute of Advanced Tech., CAS 
Address: 1068 Xueyuan Blvd., Shenzhen, Guangdong, China
Postcode: 518055
jovialniyo93@gmail.com

## License ##
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

