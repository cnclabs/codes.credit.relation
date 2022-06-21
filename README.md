# Default Prediction Research Project

**Project Description**

Employ machine learning techniques to estimate the **term structure of cumulative default probabilities** — a structured estimation that contains default probabilities from short-term to long-term periods (E.g., 1, 3, 6, 12, ..., 60 months)

**Current Implemented Models**

* Multiperiod Corporate Default Prediction—A Forward Intensity Approach (FIM)
  * Journal of Econometrics, 2012
  * Paper Link: https://www.sciencedirect.com/science/article/pii/S0304407612001145
* Multiperiod Corporate Default Prediction Through Neural Parametric Family Learning
  * Proceedings of the 2022 SIAM International Conference on Data Mining (SDM)
  * Paper Link: https://epubs.siam.org/doi/abs/10.1137/1.9781611977172.36
  * Neural Network Models: MLP, RNNs: LSTM & GRU

# Environment

Neural network models have been implemented by two different ML frameworks: PyTorch & TensorFlow 

**PyTorch Version**
* Python 3.8
* PyTorch 1.10.2

**TensorFlow Version**
* Python 3.7
* TensorFlow 1.15.0

Other Packages: Check the **requirements.txt** file in the corresponding folder (PyTorch_Ver & TensorFlow_Ver)

# How to Run the Program

First, go to the data directory and execute the get_data.sh script to download data (Refer to Dataset Description for more information)

```
$ cd data
# Make sure you are on server cfda4
$ ./get_data.sh
```

Second, choose which version of code you want to use: PyTorch or TensorFlow

And then go to the corresponding **run_models_scripts** directory

Finally, execute the corresponding bash script of the model you want to run

**PyTorch Version**: run the cross-time GRU model for example
```
$ cd PyTorch_Ver
$ cd run_models_scripts
$ ./run_gru_time.sh
```

**TensorFlow Version**: run the cross-sectional LSTM model for example
```
$ cd TensorFlow_Ver
$ cd run_models_scripts
$ ./run_lstm_index.sh
```
Note: Just ignore the tf warnings shown on the terminal

# Expected Results

# Data Description
**Dataset Used**

A real-world default and bankruptcy dataset provided by **CRI**, which is publicly available and contains 1.5 million monthly samples of US public companies over the period from **January 1990 to December 2017**

* CRI Official Website: https://nuscri.org/en/home/
* NUS Credit Research Initiative Technical Report: https://d.nuscri.org/static/pdf/Technical%20report_2020.pdf

**Data Path on Lab Server**
* Complete processed data for above mentioned implemented models: cfda4:/tmp2/cywu/default_cumulative/data/processed
* cfdaAlpha:/tmp2/cywu/default_cumulative/data

# Ideas & Project Plan
**Main Idea**

Develop a machine learning algorithm that can incorporate the **relational information** between US public companies into the model to make **correlated default predictions**

**Project Plan (To-do list)**
1. Apply ADGAT model to the CRI dataset
  * ADGAT Model
    * Modeling the Momentum Spillover Effect for Stock Prediction via Attribute-Driven Graph Attention Networks (AAAI, 2021)
    * Paper Link: https://ojs.aaai.org/index.php/AAAI/article/view/16077
    * Official Released Code: https://github.com/RuichengFIC/ADGAT
2. Design a more explainable model
3. Find & implement other suitable term structure baseline default prediction models in Finance area

# Other Relevant Works
* Multi-period Corporate Default Prediction with Stochastic Covariates
  * Journal of Financial Economics, 2007
  * Paper Link: https://www.gsb.stanford.edu/sites/default/files/publication-pdf/1-s2.0-s0304405x06002029-main.pdf


