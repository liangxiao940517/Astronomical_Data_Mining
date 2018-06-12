#程序段1
import numpy as np
import pandas as pd
import os
from pandas.core.frame import DataFrame

#对训练数据进行抽样
map_type = {'star': 0, 'unknown': 1, 'galaxy': 2, 'qso': 3}
train_index_label = pd.read_csv('F:/Python/Tianwenshuju/second_stage/second_a_train_index_20180313.csv')
# train_index_label = train_index_label[train_index_label['type'].isin(['galaxy','qso'])]
train_data_path = 'F:/Python/Tianwenshuju/second_stage/train_data/'
train_index_label['type'] = train_index_label['type'].map(map_type)
# train_index_label_pd = {train_index_label[['id']]:train_index_label[['type']]}
# print(trian_index_label_pd)

# train_index.info()#483851
train_index_label_sampled = train_index_label
train_index_label_sampled = train_index_label.sample(frac = 0.03, replace = False, axis = 0)
print(train_index_label_sampled['type'].value_counts())
train_index_label_sampled_np = train_index_label_sampled.as_matrix()
print(train_index_label_sampled_np)
# print(train_index_label_sampled)
train_index_sampled = train_index_label_sampled['id']
train_index_sampled_np = train_index_sampled.as_matrix()
print(train_index_sampled_np.shape[0])
train_label_sampled = train_index_label_sampled['type']
train_label_sampled_np = train_label_sampled.as_matrix()
# print(train_label_sampled_np['type'].info)

count = 0
train_data_pd = pd.DataFrame(np.zeros((train_index_sampled_np.shape[0], 2602)))
for index, filename_str in enumerate(train_index_sampled_np):
#     print(filename_str.dtype)
    filename = train_data_path + str(filename_str) + '.txt'
#     data_train = pd.read_csv(filename)
    file = open(filename)
    data_train = file.read()
    data_train = data_train.split(',')
    data_train = DataFrame(data_train).T
    train_data_pd.iloc[count,2600] = filename_str
    train_data_pd.iloc[count,0:2600] = data_train.iloc[0,:]
    train_data_pd.iloc[count,2601] = train_label_sampled_np[count]
    count = count + 1
    print(count)
print(train_data_pd.head())
train_data_pd.to_csv('F:/Python/Tianwenshuju/second_stage/train_data_sampled_003_5.csv')
# train_label_sampled_np = 
# train_index_sampled.info()

# #对测试数据进行抽样
# test_index_label = pd.read_csv('D:/Python_Data/tianwen/first_test_index_20180131.csv')
# test_data_path = 'D:/Python_Data/tianwen/test_data/'
# # test_index_label['type'] = test_index_label['type'].map(map_type)
# # test_index_label_sampled = test_index_label.sample(frac = 0.01, replace = False, axis = 0)
# test_index_label_sampled = test_index_label
# # test_index_label_sampled_np = test_index_label_sampled.as_matrix()
# test_index_sampled = test_index_label_sampled['id']
# test_index_sampled_np = test_index_sampled.as_matrix()
# # test_label_sampled = test_index_label_sampled['type']
# # test_label_sampled_np = test_label_sampled.as_matrix()

# count1 = 0
# test_data_pd = pd.DataFrame(np.zeros((test_index_sampled_np.shape[0], 2601)))
# for test_index, test_filename_str in enumerate(test_index_sampled_np):
#     test_filename = test_data_path + str(test_filename_str) + '.txt'
#     test_file = open(test_filename)
#     data_test = test_file.read()
#     data_test = data_test.split(',')
#     data_test = DataFrame(data_test).T
#     test_data_pd.iloc[count1, 0:2600] = data_test.iloc[0,:]
#     test_data_pd.iloc[count1, 2600] = test_filename_str
#     count1 = count1 + 1
#     print(count1)
# print(test_data_pd.head())
# test_data_pd.to_csv('I:/Python/Tianwen/test_data_0.01.csv')
