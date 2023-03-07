import pandas as pd

# model_dir = f'/tmp2/cwlin/default_prediction/Default_Prediction_Models/experiments/gru{window}_index'
data_dir = f'/tmp2/cwlin/default_prediction/Default_Prediction_Models/data/8_labels_index/len_{window}/index_fold_{fold}'
output_all_company_ids_dir = '/tmp2/cwlin/explainable_credit/data'
label_type = 'cum'

train_data = "train_{}.gz".format(label_type)
test_data  = "test_{}.gz".format(label_type)

train_data = os.path.join(data_dir, train_data)
eval_data  = os.path.join(data_dir, test_data)

filename = train_data
feature_size = 14
window_size = int(window)

label_type = 'cum'
train_data = "train_{}.gz".format(label_type)
test_data  = "test_{}.gz".format(label_type)
filename = train_data
compression = "gzip" if ".gz" in filename else None

data_stats = dict()

window = '06'
fold = '06'
data_dir = f'/tmp2/cwlin/default_prediction/Default_Prediction_Models/data/8_labels_index/len_{window}/index_fold_{fold}'
output_dir = f'/tmp2/cwlin/default_prediction/Default_Prediction_Models/data/8_labels_index/len_{window}/index_fold_{fold}'
train_path = os.path.join(data_dir, train_data)
test_path  = os.path.join(data_dir, test_data)

train_all_df = pd.read_csv(train_path, compression=compression, header=0)
train_all_company = train_all_df.id.sort_values().unique()

test_all_df = pd.read_csv(test_path, compression=compression, header=0)
test_all_company = test_all_df.id.sort_values().unique()

data_stats[(window, fold)] = [train_all_company, test_all_company]

train, test, = data_stats[('06', '06')]
pd.DataFrame(list(train), columns=['id']).to_csv(f'{output_all_company_ids_dir}/all_company_ids.csv')