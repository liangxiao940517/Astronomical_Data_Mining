import pandas as pd
import numpy as np
import tensorflow as tf
import os
# #训练集
# data = pd.read_csv('G:/Python/Tianwen/first_train_index_20180131.csv')
# map_type = {'star': 0, 'unknown': 1, 'galaxy': 2, 'qso': 3}
# data['type'] = data['type'].map(map_type)
# data_type_np = data['type'].as_matrix()
# print(data_type_np)
# data_id_np = data['id'].as_matrix()
# print(data_id_np)

# file_path = 'G:/Python/Tianwen/train_data/'
# # train_file = open('F:/Python/Tianwenshuju/train_data/683853.txt')
# datafile_path = []
# label = []
# # train = train_file.read()
# # train = train.split(',')
# for index,filename in enumerate(data_id_np):
#     filepath = file_path + str(filename) + '.txt'
#     datafile_path.append(filepath)
# #     label.append(data[data['id'] == filename]['type'])
#     label.append(data_type_np[index])
#     print(index)
# print(datafile_path[0])

# #开发集
# dev_data = pd.read_csv('G:/Python/Tianwen/dev_data.csv')
# dev_data_id_np = dev_data['id'].as_matrix()
# # dev_data['type'] = dev_data['type'].map(map_type)
# dev_data_type_np = dev_data['type'].as_matrix()
# dev_path = 'G:/Python/Tianwen/dev_data/'
# dev_file_path = []
# dev_label = []
# for index, filename in enumerate(dev_data_id_np):
#     devpath = dev_path + str(filename) + '.txt'
#     dev_file_path.append(devpath)
#     dev_label.append(dev_data_type_np[index])


# #测试集
# predict = pd.read_csv('G:/Python/Tianwen/first_test_index_20180131.csv')
# predict_id_np = predict['id'].as_matrix()
# print(predict_id_np)
# predict_file_path = 'G:/Python/Tianwen/test_data/'
# predict_data_file_path = []
# for index, filename in enumerate(predict_id_np):
#     predictpath = predict_file_path + str(filename) + '.txt'
#     predict_data_file_path.append(predictpath)


dropout1 = 0.5
learning_rate = 0.1
num_classes = 4
train_filenames = datafile_path
dev_filenames = dev_file_path
predict_filenames = predict_data_file_path
train_labels = label
dev_labels = dev_label
def _parse_function(filename, label):
    data_train = tf.read_file(filename)
#     data_train = data_train.read()
    data_train = tf.string_split([data_train], delimiter = ',')
    data_train = data_train.values
    data_train = tf.string_to_number(data_train, tf.float64)
    features = data_train
#     data_train = data_train.split(',')
#     features = tf.constant(data_train)
    print(label)
#     label = tf.one_hot(label, depth = num_classes)
    return features, label
def _predict_parse_function(filename):
    data_train = tf.read_file(filename)
#     data_train = data_train.read()
    data_train = tf.string_split([data_train], delimiter = ',')
    data_train = data_train.values
    data_train = tf.string_to_number(data_train, tf.float64)
    features = data_train
    return features
def train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    dataset = dataset.shuffle(483851)
    dataset = dataset.map(_parse_function)
    dataset = dataset.repeat(2)
    dataset = dataset.batch(50)
    iterator = dataset.make_one_shot_iterator()
    train_feature, train_label = iterator.get_next()
#     train_label = tf.one_hot(train_label, num_classes)
#     print(train_label)
    return train_feature, train_label

def dev_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((dev_filenames, dev_labels))
    dataset = dataset.map(_parse_function)
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    dev_feature, dev_label = iterator.get_next()
    return dev_feature, dev_label

def predict_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices((predict_filenames))
    dataset = dataset.map(_predict_parse_function)
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    predict_feature = iterator.get_next()
    return predict_feature

def conv_net(features, n_classes, dropout, reuse, is_training):
    features = tf.reshape(features, shape = [-1,2600])
    features = tf.layers.batch_normalization(inputs = features)
    print(features.shape)
    fc1 = tf.layers.dense(features, 256, activation = tf.nn.relu)
#     fc1 = tf.layers.dropout(fc1, rate = dropout, training = is_training)
    
    fc1 = tf.layers.dense(fc1, 128, activation = tf.nn.relu)
#     fc1 = tf.layers.dropout(fc1, rate = dropout, training = is_training)
    
    fc1 = tf.layers.dense(fc1, 128, activation = tf.nn.relu)
#     fc1 = tf.layers.dropout(fc1, rate = dropout, training = is_training)
    
    out = tf.layers.dense(fc1, n_classes)
    return out

def model_fn(features, labels, mode):
    logits_train = conv_net(features, num_classes, dropout1, reuse = False, is_training = True)
    logits_dev = conv_net(features, num_classes, dropout1, reuse = True, is_training = False)
    pred_classes = tf.argmax(logits_dev, axis = 1)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions = pred_classes)
    
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits_train, labels = tf.cast(labels, dtype = tf.int32)))
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_op = optimizer.minimize(loss_op, global_step = tf.train.get_global_step())
    
    acc_op = tf.metrics.accuracy(labels = labels, predictions = pred_classes)
    
    estim_specs = tf.estimator.EstimatorSpec(mode = mode, predictions = pred_classes, loss = loss_op, train_op = train_op, eval_metric_ops = {'accuracy': acc_op})
    
    return estim_specs

model = tf.estimator.Estimator(model_fn)
model.train(train_input_fn, steps = 20000)

evaluate = model.evaluate(dev_input_fn)
print("Testing Accuracy:", evaluate['accuracy'])

predict = model.predict(predict_input_fn)
