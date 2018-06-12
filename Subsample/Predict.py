#对开发数据进行评测
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import os

dev_index_label = pd.read_csv('G:/Python/Tianwen/dev_set.csv')
dev_data_path = 'G:/Python/Tianwen/train_data/'
# test_index_label = pd.read_csv('D:/Python_Data/tianwen/first_test_index_20180131.csv')
# test_data_path = 'D:/Python_Data/tianwen/test_data/'
# test_index_label['type'] = test_index_label['type'].map(map_type)
# test_index_label_sampled = test_index_label.sample(frac = 0.01, replace = False, axis = 0)
# test_index_label_sampled_np = test_index_label_sampled.as_matrix()
dev_index = dev_index_label['id']
dev_index_np = dev_index.as_matrix()
mapmap = {'star':0, 'unknown':1, 'galaxy':2,'qso':3}
dev_label = dev_index_label['type'].map(mapmap)
dev_label_np = dev_label.as_matrix()

# test_label_sampled = test_index_label_sampled['type']
# test_label_sampled_np = test_label_sampled.as_matrix()


# print(dirs)

count2 = 0
dev_data_pd = pd.DataFrame(np.zeros((dev_index_np.shape[0], 2)), columns = ['id','type'])
for dev_index, dev_filename_str in enumerate(dev_index_np):
    dev_filename = dev_data_path + str(dev_filename_str) + '.txt'
    dev_file = open(dev_filename)
    data_dev = dev_file.read()
    data_dev = data_dev.split(',')
    data_dev = DataFrame(data_dev).T
    y_predict_dev = bgc.predict(data_dev)
    dev_data_pd.iloc[count2,0] = dev_filename_str
    dev_data_pd.iloc[count2,1] = y_predict_dev
    print(y_predict_dev)
#     test_data_pd.iloc[count1, 0:2600] = data_test.iloc[0,:]
#     test_data_pd.iloc[count1, 2600] = test_filename_str
    count2 = count2 + 1
    print(count2)
print(predict_data_pd.head())
dev_data_np = dev_data_pd['type'].as_matrix()
aaaa = f1_score(dev_label_np, dev_data_np, average = 'macro')
print("Macro F1 Score: %f" % aaaa)
#GradientBoosting_2000:0.650929,线上:0.58
#加样本之后GradientBoosting_2000:0.779988,线上:0.68
#加样本之后GradientBoosting_3000:0.779790
#加了两次galaxy和qso之后GradientBoosting_2000: 0.764973
