# Default Prediction Models Implementation

**How to run the program**

First, go to the data directory and execute the get_data.sh script to download data

```
$ cd data
$ ./get_data.sh
```

Second, choose which version of code you want to use: PyTorch or TensorFlow

And then go to the corresponding run_models_scripts directory

Finally, execute the corresponding bash script of the model you want to run

**PyTorch Version**: run the cross-time GRU model
```
$ cd PyTorch_Ver
$ cd run_models_scripts
$ ./run_gru_time.sh
```

**TensorFlow Version**: run the cross-sectional LSTM model
```
$ cd TensorFlow_Ver
$ cd run_models_scripts
$ ./run_lstm_index.sh
```
