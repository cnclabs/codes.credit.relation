# Default Prediction Research Project

**Project Description**

Employ and design an **explainable** model to estimate the **term structure of cumulative default probabilities** — a structured estimation that contains default probabilities from short-term to long-term periods (E.g., 1, 3, 6, 12, ..., 60 months)

**Previous Works**

* Multiperiod Corporate Default Prediction—A Forward Intensity Approach (FIM)
  * Journal of Econometrics, 2012
  * Paper Link: https://www.sciencedirect.com/science/article/pii/S0304407612001145
* Multiperiod Corporate Default Prediction Through Neural Parametric Family Learning
  * Proceedings of the 2022 SIAM International Conference on Data Mining (SDM)
  * Paper Link: https://epubs.siam.org/doi/abs/10.1137/1.9781611977172.36
  * Neural Network Models: MLP, RNNs: LSTM & GRU

**Current Implemented Models**

* ADGAT Model
  * Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-Driven Graph Attention Networks (AAAI, 2021)
  * Paper Link: https://ojs.aaai.org/index.php/AAAI/article/view/16077
  * Official Released Code: https://github.com/RuichengFIC/ADGAT

**Revised Part**
  * To enhance the accuracy of the predicted results with relational information, we will incorporate the ADGAT model.
  * To meet the model's requirements, the dataset loader output will be adapted to accommodate multiple companies.
  * To reduce GPU memory usage, we have implemented a data sampling function.
  * Import the early stop function and set the patience parameter.
  * To align with our proposed approach, we will modify the Graph_GRUCell model in Layer.py to enable weight value sharing.

# Environment

Neural network models have been implemented by two different ML frameworks: PyTorch & TensorFlow 

**PyTorch Version**
* Python 3.8
* PyTorch 1.11.0

Other Packages: Check the **requirements.txt** file in the corresponding folder

# How to Run the Program

First, go to the data directory and execute the get_data.sh script to download complete data (Refer to Dataset Description for more information)
(warning : the full dataset cannot be loaded in the model due to the limit of CUDA memory)
```
$ cd codes/data
# Make sure you are on server cfda4
$ bash get_data.sh
```
or create the sample data from following steps
```
$ cd codes/build_sampling/preprocess
$ bash sampling.sh
```

Second, choose which dataset you want to use: cross-section or cross-time

And then go to the corresponding **run_models_scripts** directory

Finally, execute the corresponding bash script of the model you want to run

**Cross-section** : run on the cross-sction dataset
```
$ cd codes/run_models_scripts
$ bash ./run_gru_index.sh
```
**Cross-time** : run on the cross-time dataset
```
$ cd codes/run_models_scripts
$ bash ./run_gru_time.sh
```
Note: Just ignore the tf warnings shown on the terminal

# Expected Results
Reproduce the results of Tables 1 & 2 of the paper: Multiperiod Corporate Default Prediction Through Neural Parametric Family Learning

**Results generated by the code in the repo (Pending update...)**: Collect results.csv **average** column: cap -> AR & recall -> RMSNE


# Data Description
**Dataset Used**

A real-world default and bankruptcy dataset provided by **CRI**, which is publicly available and contains 1.5 million monthly samples of US public companies over the period from **January 1990 to December 2017**

* CRI Official Website: https://nuscri.org/en/home/
* NUS Credit Research Initiative Technical Report: https://d.nuscri.org/static/pdf/Technical%20report_2020.pdf
* Initially preprocessed data: merged.csv

**Data Overview**
* Check file **Check_Data.ipynb**'s contents (Notice that the read file path will be different if you want to execute the code on the server)

**Data Path on Lab Server**
* CRI raw data: **cfdaAlpha**:/tmp2/cywu/default_cumulative/data
  * Unzip nus_raw.tar -> Get raw_data folder
    * Files: Company_Mapping.csv, File_Location_by_Firms.csv, US_Firms_Specific
  * Initially preprocessed data (Refer to code files in Default_Prediction_Models/TensorFlow_Ver/preprocess/)
    * Files in folder interim 
* Complete processed data for above mentioned implemented models: **cfda4**:/tmp2/cywu/default_cumulative/data/processed
  * 8_labels_index: Cross-sectional Experiment
  * 8_labels_time: Cross-time Experiment

# Current Results
We anticipate that the inclusion of the ADGAT relation model will enhance the results with an explanation that can be understood. After fine-tuning the parameters, we achieved the following optimal outcomes

## The SDM model without relation vs SDM model with relation

* We sample 500 companies with cross-section dataset
* adgat=incorporate agdat relation model
* hidden size = {32, 64, 128}
* learning rate = {1e-03, 1e-04, 1e-05}
* weight decay = {1e-04, 1e-05, 1e-06}
* Number of Layer = 1
* batch size = {1, 5, 10, 15}
* patience = {20, 50, 75, 100}
* max epoch = 300

AR :
| ADGAT relation | Window size | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| False | 1 | 92.01% | 92.98% | 92.90% | 92.61% | 88.89% | 85.97% | 82.62% | 77.39% |
| False | 6 | 93.68% | 94.60% | 94.56% | 93.21% | 91.53% | 89.45% | 87.12% | 83.52% |
| False | 12 | 96.57% | 98.01% | 97.91% | 96.71% | 96.11% | 94.75% | 93.95% | 93.72% |
| True | 1 | 88.65% | 89.73% | 87.67% | 87.46% | 81.58% | 76.82% | 74.09% | 69.72% |
| True | 6 | 92.08% |	94.93% |	94.37% |	93.49% |	92.50% |	90.49% |	88.89% |	84.96% |
| True | 12 | 92.73% |	97.70% |	97.44% |	96.71% |	95.97% |	94.45% |	93.88% |	93.21% |

RMSNE :
| ADGAT relation | Window size | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| False | 1 | 0.87 | 0.79 | 0.71 | 0.70 | 0.62 | 0.64 | 0.63 | 0.64 |
| False | 6 | 0.84 | 0.75 | 0.69 | 0.66 | 0.64 | 0.63 | 0.64 | 0.64 |
| False | 12 | 0.70 | 0.58 | 0.64 | 0.61 | 0.58 | 0.58 | 0.58 | 0.59 |
| True | 1 | 0.94 |	0.88 |	0.82 |	0.76 |	0.72 |	0.69 |	0.62 |	0.49 |
| True | 6 | 0.92 |	0.84 |	0.77 |	0.65 |	0.62 |	0.63 |	0.65 |	0.65 |
| True | 12 | 0.81 |	0.67 |	0.61 |	0.57 |	0.58 |	0.57 |	0.57 |	0.58 |

# Other Relevant Works
* Multi-period Corporate Default Prediction with Stochastic Covariates
  * Journal of Financial Economics, 2007
  * Paper Link: https://www.gsb.stanford.edu/sites/default/files/publication-pdf/1-s2.0-s0304405x06002029-main.pdf
