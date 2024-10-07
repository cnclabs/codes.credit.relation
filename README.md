# Time-aware Graph Attention Networks (TAGAT) for Multiperiod Default Prediction

This repository contains the implementation of the Time-aware Graph Attention Networks (TAGAT) for multiperiod default prediction in credit risk management, as presented in our paper "Time-aware Graph Attention Networks for Multiperiod Default Prediction".

## Model architecture


## Usage 
### Running the Scripts
 quickly start running our TAGAT model, you can use the provided shell script `run_GAT.sh`. 

```
./run_GAT.sh <device> <fold>
```

Replace `<device>` with the device identifier (e.g., `cpu` or the index of your cuda device) and `<fold>` with the specific fold number within the data folder (e.g., 1, 2, etc.).

### Python Files Explanation
The shell script `run_GAT.sh` utilizes several Python files to train, validate, and evaluate the TAGAT model. Here's an explanation of the purpose of each Python file:

#### `main_valid.py`
This script is responsible for training and validating the model. It performs hyperparameter tuning and early stopping based on validation performance.
* **Purpose**: Find the optimal number of epochs based on validation data. 
* **Key Arguments**:
  * `--model_nam`: Name of the model (e.g., NeuDP_GAT).
  * `--device`: Device to run the model on (e.g., cpu or the index of your cuda device).
  * `--experiment_type`: Type of how the data is preprocessed (e.g., expand_len).
  * `--fold`: Fold number of the data.
  * `--max_epoch`: Maximum number of epochs for training. 
  * `--patience`: Patience for early stopping.
  * `--batch_size`: Batch size for training. In our case, it indicates how many timestamp of data are utilized during the training.
  * `--learning_rate`: Learning rate for the optimizer.
  * `--weight_decay`: Weight decay (L2 regularization).
  * `--gamma`: Exponential learning rate scheduler.
  * `--lstm_num_units`: Number of units in the RNN layer.
  * `--cluster_setting`: Clustering setting (e.g., industry).
  * `--n_cluster`: Number of clusters, which is the number of sectors in our case.
  * `--intra_gat_hidn_dim`: Hidden dimension for intra-sector GAT.
  * `--inter_gat_hidn_dim`: Hidden dimension for inter-sector GAT.

#### `main.py`
This script is responsible for training the model using the number of epochs determined during the validation phase.

* **Purpose**: Train the model using the full training set and the optimal number of epochs.
* **Key Arguments**: Same as main_valid.py, with the addition of:
  * `--num_epochs`: Number of epochs to train the model.


#### `predict.py`
This script is used for generating predictions on the test dataset using the trained model.

* **Purpose**: Generate predictions for the test dataset.
* **Key Arguments**: Same as `main.py`, with the addition of:
  * `--data_file`: The file containing the test data. 
  * `--restore_dir`: Directory from which to restore the trained model weights.

#### `report.py`
This script generates a report comparing the predicted values with the actual values to evaluate model performance.

* **Key Arguments**: Same as `main.py`, with the addition of:
  * `--pred_file`: File containing the model's predictions. 
  * `--label_file`: File containing the ground truth labels.

#### Generating Evaluation Metrics
The following commands in the shell script extract the Accuracy Ratio (AR) and Root Mean Square Normalized Error (RMSNE) metrics from the results:

```sh
python -c "import json; print('\n'.join(str(v) for k, v in json.load(open('$model_dir/test_metrics_pred_best_weights.json')).items() if k.startswith('cap')))" >> $model_dir/AR &&
python -c "import json; print('\n'.join(str(v) for k, v in json.load(open('$model_dir/test_metrics_pred_best_weights.json')).items() if k.startswith('recall')))" >> $model_dir/RMSNE  
```

* **Purpose**: Extract and save the AR and RMSNE metrics from the JSON file containing the evaluation results.