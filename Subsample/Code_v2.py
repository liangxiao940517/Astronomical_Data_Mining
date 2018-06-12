#程序段2
import numpy as np
import pandas as pd
import os
from pandas.core.frame import DataFrame
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_selection import SelectKBest


train_data_dataframe_003_0217 = pd.read_csv('F:/Python/Tianwenshuju/train_data_0.03_0217.csv')
print(train_data_dataframe_003_0217.shape)
train_data_dataframe_003_0220 = pd.read_csv('F:/Python/Tianwenshuju/train_data_0.03_0217_2.csv')
print(train_data_dataframe_003_0220.shape)
train_data_dataframe_005_0217 = pd.read_csv('F:/Python/Tianwenshuju/train_data_0.05_0217.csv')
print(train_data_dataframe_005_0217.shape)
train_data_dataframe_005_0220 = pd.read_csv('F:/Python/Tianwenshuju/train_data_0.05_0220.csv')
print(train_data_dataframe_005_0220.shape)
pieces = [train_data_dataframe_003_0217, train_data_dataframe_003_0220, train_data_dataframe_005_0217, train_data_dataframe_005_0220]
train_data_dataframe = pd.concat(pieces)
print(train_data_dataframe.shape)
print(train_data_dataframe.head())
train_data_dataframe.drop(['Unnamed: 0'], inplace = True, axis = 1)
print(train_data_dataframe)
# print(train_data_dataframe.head())
# test_data_dataframe = pd.read_csv('I:/Python/Tianwen/test_data.csv')


train_data_numpy = train_data_dataframe.as_matrix()
# test_data_numpy = test_data_dataframe.as_matrix()
# print(train_data_dataframe)
# print(train_data_numpy)
X_train = train_data_numpy[:, 0:2600]
y_train = train_data_numpy[:, 2601]
# X_predict = test_data_numpy[:, 0:2600]
bgc = BaggingClassifier(n_estimators = 100, max_features = 0.5)
bgc.fit(X_train, y_train)
# print(X_train.shape)
# print(X_predict.shape)
# y_predict = bgc.predict(X_predict)
# print(y_predict)



#对测试数据进行抽样
test_index_label = pd.read_csv('F:/Python/Tianwenshuju/first_test_index_20180131.csv')
test_data_path = 'F:/Python/Tianwenshuju/test_data/'
# test_index_label = pd.read_csv('D:/Python_Data/tianwen/first_test_index_20180131.csv')
# test_data_path = 'D:/Python_Data/tianwen/test_data/'
# test_index_label['type'] = test_index_label['type'].map(map_type)
# test_index_label_sampled = test_index_label.sample(frac = 0.01, replace = False, axis = 0)
test_index_label_sampled = test_index_label
# test_index_label_sampled_np = test_index_label_sampled.as_matrix()
test_index_sampled = test_index_label_sampled['id']
test_index_sampled_np = test_index_sampled.as_matrix()
# test_label_sampled = test_index_label_sampled['type']
# test_label_sampled_np = test_label_sampled.as_matrix()

count1 = 0
predict_data_pd = pd.DataFrame(np.zeros((test_index_sampled_np.shape[0], 2)), columns = ['id','type'])
for test_index, test_filename_str in enumerate(test_index_sampled_np):
    test_filename = test_data_path + str(test_filename_str) + '.txt'
    test_file = open(test_filename)
    data_test = test_file.read()
    data_test = data_test.split(',')
    data_test = DataFrame(data_test).T
    y_predict = bgc.predict(data_test)
    predict_data_pd.iloc[count1,0] = test_filename_str
    predict_data_pd.iloc[count1,1] = y_predict
    print(y_predict)
#     test_data_pd.iloc[count1, 0:2600] = data_test.iloc[0,:]
#     test_data_pd.iloc[count1, 2600] = test_filename_str
    count1 = count1 + 1
    print(count1)
print(predict_data_pd.head())
map_result = {0:'star', 1:'unknown', 2:'galaxy',3:'qso'}
predict_data_pd['type'] = predict_data_pd['type'].map(map_result)
predict_data_pd.to_csv('F:/Python/Tianwenshuju/predict_data_0226.csv')
