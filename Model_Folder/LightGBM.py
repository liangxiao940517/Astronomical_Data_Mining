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
import lightgbm as lgb



train_data_dataframe_003_1 = pd.read_csv('F:/Python/Tianwenshuju/second_stage/train_data_sampled_003_1.csv')
train_data_dataframe_003_2 = pd.read_csv('F:/Python/Tianwenshuju/second_stage/train_data_sampled_003_2.csv')
train_data_dataframe_003_3 = pd.read_csv('F:/Python/Tianwenshuju/second_stage/train_data_sampled_003_3.csv')
train_data_dataframe_003_4 = pd.read_csv('F:/Python/Tianwenshuju/second_stage/train_data_sampled_003_4.csv')
train_data_dataframe_003_5 = pd.read_csv('F:/Python/Tianwenshuju/second_stage/train_data_sampled_003_5.csv')
# train_data_dataframe_005_0220 = pd.read_csv('F:/Python/Tianwenshuju/train_data_0.05_0220.csv')
train_data_dataframe_galaxy_qso = pd.read_csv('F:/Python/Tianwenshuju/second_stage/train_data_galaxy_qso.csv')
# train_data_galaxy_qso = pd.read_csv('F:/Python/Tianwenshuju/second_stage/train_data_galaxy_qso.csv')
pieces = [train_data_dataframe_003_1, train_data_dataframe_003_2, train_data_dataframe_003_3, train_data_dataframe_003_4, train_data_dataframe_003_5, train_data_dataframe_galaxy_qso]
train_data_dataframe = pd.concat(pieces)
print(train_data_dataframe.shape)

# train_data_dataframe_ = pd.read_csv('G:/Python/Tianwen/train_data_0.05_0220.csv')
train_data_dataframe.drop(['Unnamed: 0'], inplace = True, axis = 1)
train_data_dataframe = train_data_dataframe.sample(frac = 1)
print(train_data_dataframe)
# print(train_data_dataframe.head())
# test_data_dataframe = pd.read_csv('I:/Python/Tianwen/test_data.csv')
# train_data_dataframe = pd.read_csv('/data/')

train_data_numpy = train_data_dataframe.as_matrix()
# test_data_numpy = test_data_dataframe.as_matrix()
# print(train_data_dataframe)
# print(train_data_numpy)
X_train = train_data_numpy[:, 0:2600]
y_train = train_data_numpy[:, 2601]

lgb_train = lgb.Dataset(X_train, label = y_train)
# X_predict = test_data_numpy[:, 0:2600]
# bgc = BaggingClassifier(n_estimators = 10, max_features = 1.0)
# bgc = RandomForestClassifier(n_estimators = 10, max_features = 1.0, random_state = 2048)
# bgc = ExtraTreesClassifier(n_estimators = 10, max_features = 1.0, random_state = 2048)
# bgc = GradientBoostingClassifier(n_estimators = 2000, max_features = 'sqrt', random_state = 2048)
# bgc = AdaBoostClassifier(n_estimators = 250, random_state = 2048)

lgb_paramas = {'task':'train',
               'boosting':'gbdt',# 这已经是默认的参数设置了
               'application':'multiclass',
               'num_class':4,
                'meric':'multi_logloss',
#                'min_data_in_leaf':500,
                 'num_leaves':31,
                 'learning_rate':0.05,
                 'feature_fraction':1.0,
                 'bagging_feaction':1.0,
                 'bagging_freq':2,
               'bagging_seed':5048,
               'feature_fraction_seed':2048
}
print('start training')
bst = lgb.train(lgb_paramas, lgb_train, num_boost_round = 1000)
print('finished')
# bgc.fit(X_train, y_train)
# print(X_train.shape)
# print(X_predict.shape)
# y_predict = bgc.predict(X_predict)
# print(y_predict)



#对测试数据进行抽样
test_index_label = pd.read_csv('F:/Python/Tianwenshuju/second_stage/second_a_test_index_20180313.csv')
test_data_path = 'F:/Python/Tianwenshuju/second_stage/test_data/'
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
    print(data_test_np)
#     y_predict = bgc.predict(data_test)
    y_predict = bst.predict(data_test_np)
    y_predict = y_predict.tolist()
    y_predict = y_predict[0].index(max(y_predict[0]))
    print(y_predict)
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
# predict_data_pd.to_csv('G:/Python/Tianwen/predict_data_0.16_LightGBM_1000_seed2048.csv')
predict_data_pd.to_csv('F:/Python/Tianwenshuju/second_stage/submit_file_326.csv')






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
    data_dev_np = data_dev.as_matrix()
    y_predict_dev = bst.predict(data_dev_np)
    print(y_predict_dev)
    y_predict_dev = y_predict_dev.tolist()
    y_predict_dev = y_predict_dev[0].index(max(y_predict_dev[0]))
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
#LightGBM1000次迭代，num_leaves为31，线下：0.794325 线上：0.7
#LightGBM1000次迭代，num_leaves为127，线下：0.785669 线上：0.69
