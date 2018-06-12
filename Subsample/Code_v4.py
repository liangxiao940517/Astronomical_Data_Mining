import numpy as np
import pandas as pd
import os
from pandas.core.frame import DataFrame
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


train_data_dataframe_003_0217 = pd.read_csv('F:/Python/Tianwenshuju/train_data_0.03_0217.csv')
train_data_dataframe_003_0220 = pd.read_csv('F:/Python/Tianwenshuju/train_data_0.03_0217_2.csv')
train_data_dataframe_005_0217 = pd.read_csv('F:/Python/Tianwenshuju/train_data_0.05_0217.csv')
train_data_dataframe_005_0220 = pd.read_csv('F:/Python/Tianwenshuju/train_data_0.05_0220.csv')
train_data_dataframe_galaxy_qso = pd.read_csv('F:/Python/Tianwenshuju/train_data_galaxy_qso.csv')
pieces = [train_data_dataframe_003_0217, train_data_dataframe_003_0220, train_data_dataframe_005_0217, train_data_dataframe_005_0220, train_data_dataframe_galaxy_qso]
train_data_dataframe = pd.concat(pieces)
print(train_data_dataframe.shape)

# train_data_dataframe_ = pd.read_csv('G:/Python/Tianwen/train_data_0.05_0220.csv')
train_data_dataframe.drop(['Unnamed: 0'], inplace = True, axis = 1)
train_data_dataframe = train_data_dataframe.sample(frac = 1)
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
# bgc = BaggingClassifier(n_estimators = 10, max_features = 1.0)
# bgc = RandomForestClassifier(n_estimators = 10, max_features = 1.0, random_state = 2048)
# bgc = ExtraTreesClassifier(n_estimators = 10, max_features = 1.0, random_state = 2048)
# bgc = GradientBoostingClassifier(n_estimators = 2000, max_features = 'sqrt', random_state = 2048)
# bgc = AdaBoostClassifier(n_estimators = 250, random_state = 2048)

xgb1 = XGBClassifier(
    learning_rate = 0.1,
    n_estimators = 1000,
    max_depth = 5,
    min_child_weight = 1,
    gamma = 0,
    subsample = 1, 
    colsample_bytree = 0.8,
    objective = 'multi:softmax',
    num_class = 4,
    scale_pos_weight = 1,
    seed = 2048)

# bgc.fit(X_train, y_train)
xgb1.fit(X_train, y_train)
# print(X_train.shape)
# print(X_predict.shape)
# y_predict = bgc.predict(X_predict)
# print(y_predict)



#对测试数据进行抽样
test_index_label = pd.read_csv('G:/Python/Tianwen/first_test_index_20180131.csv')
test_data_path = 'G:/Python/Tianwen/test_data/'
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
    data_test_np = data_test.as_matrix()
#     y_predict = bgc.predict(data_test)
    y_predict = xgb1.predict(data_test_np)
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
predict_data_pd.to_csv('G:/Python/Tianwen/predict_data_0.16_XGBoost.csv')
