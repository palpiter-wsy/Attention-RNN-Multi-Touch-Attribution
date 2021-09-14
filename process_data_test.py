'''
https://github.com/jeremite/channel-attribution-model/blob/master/process_data.py

'''

import numpy as np
import random
import tensorflow as tf
import tf_slim as slim
# from keras.utils import to_categorical
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from pandas.api.types import is_string_dtype, is_numeric_dtype
import warnings
import pandas as pd

warnings.filterwarnings("ignore")


def cat(data, field, leng, s=','):
    data[field] = data[field].map(lambda x: x.split(s))
    data[leng] = data[field].map(lambda x: len(x))


def most_fre(lt):
    return max(lt, key=lt.count)


most_fre('north')


def sca(col):
    col = [int(v) for v in col]
    return [(v - min(col)) / (max(col) - min(col)) for v in col]


def train_cats(df):
    """Change any columns of strings in a panda's dataframe to a column of
    categorical values. This applies the changes inplace.
    """
    for n, c in df.items():
        if is_string_dtype(c):
            df[n] = c.astype('category').cat.as_ordered()


def process_data(data, seq_length=20):
    # data_a = data_att.loc[:,['path_id','path','total_conversions','last_time_lapse','null_conversion']]
    data.dropna(axis=0, how='any',
                inplace=True)  # 删除了所有na的,axis=0删除包含缺失值的行，how='any'只要有缺失值出现就删除该行，inplace=True在原数据上进行操作
    # merge target variables

    # str -> list
    cat(data, 'path', "leng_path", s='>')
    cat(data, 'marketing_area', "leng_area", s=',')
    cat(data, 'tier', "leng_tier", s=',')
    cat(data, 'customer_type', "leng_type", s=',')
    # remove those path with channels less than 3

    data_new = data[(data.leng_path >= 3)]
    # 可以还原索引，重新变为默认的整型索引
    data_new = data_new.reset_index()
    # leave with the most common value
    data_new.marketing_area = data_new.marketing_area.map(lambda x: most_fre(x))
    data_new.tier = data_new.tier.map(lambda x: most_fre(x))
    data_new.customer_type = data_new.customer_type.map(lambda x: most_fre(x))
    data_new.replace('', 'NA', inplace=True)  # ？？？？？？？？前面不是删掉了缺失行吗

    # 输出12：所有y
    y = data_new.total_conversions
    # print(y)

    # 抽取训练集、测试集
    # got train and test data indices
    idx = [x for x in range(data_new.shape[0])]  # 输出矩阵的行数,以[0, 1, 2, 3]形式输出
    np.random.seed(111)
    random.shuffle(idx)  # 打乱
    tr_idx = idx[0:int(0.9 * len(idx))]
    te_idx = idx[int(0.9 * len(idx)):]
    print(idx)
    print(tr_idx)
    print(te_idx)

    # got data for time decay
    cat(data_new, 'last_time_lapse', "leng_time_lapse", s=',')
    data_new.last_time_lapse = data_new.last_time_lapse.map(lambda x: sca(x))
    pad_sequence = tf.keras.preprocessing.sequence.pad_sequences
    time_decay = pad_sequence(data_new.last_time_lapse, maxlen=seq_length, padding='pre', truncating='pre',
                              dtype='float32')
    # print('1111111')
    # print(data_new.last_time_lapse)
    # print('before reshape')
    # print(time_decay)
    # print(time_decay.shape)
    # 输出10：所有时间
    time_decay = time_decay.reshape(-1, 20, 1)
    # print('after reshape')
    # print(time_decay)
    # print(time_decay.shape)
    # 输出1：训练集时间
    time_decay_tr = time_decay[tr_idx]
    # 输出2：测试集时间
    time_decay_te = time_decay[te_idx]

    # got data for attribution
    text = data_new.path
    # print('333333333')
    # print(text)
    # print(text.type)


    # encoding
    t = Tokenizer()
    t.fit_on_texts(text)
    # vocabulary size
    vocab_size = len(t.word_index) + 1
    # print(len(t.word_index))
    # integer encode the documents
    encoded_docs = t.texts_to_sequences(text)
    # padding and truncating path data
    # 输出11：newlines.shape,17*20,pad_sequence
    newlines = pad_sequence(encoded_docs, maxlen=seq_length, padding='pre', truncating='post')
    # print('4444444444')
    # print(newlines)
    # print(newlines.shape)
    X_train = newlines[tr_idx]
    # print('101010101010')
    # print(X_train.shape)
    # 输出7
    Y_train = y[tr_idx]
    # print(Y_train)
    # print(type(Y_train))
    X_test = newlines[te_idx]
    # 输出8
    Y_test = y[te_idx]
    # 输出9：所有x
    all_X = np.array(list(map(lambda x: np_utils.to_categorical(x, num_classes=vocab_size), newlines)), ndmin=3)
    # print('77777777')
    # print(all_X)
    m_all, _, _ = all_X.shape
    # print(m_all)
    # 输出3：用户路径模块-训练集X
    X_tr = np.array(list(map(lambda x: np_utils.to_categorical(x, num_classes=vocab_size), X_train)), ndmin=3)
    # print('8888888888')
    # print(X_tr)
    # print(X_tr.shape)
    # 输出4：用户路径模块-测试集X
    X_te = np.array(list(map(lambda x: np_utils.to_categorical(x, num_classes=vocab_size), X_test)), ndmin=3)
    # print('99999999999')
    # print(X_te)
    # print(X_te.shape)
    # 输出14：所有路径
    # print(text)
    paths = text[tr_idx].reset_index().path
    # print(text[tr_idx])
    # print(paths)

    # got customer data (control data)
    data_lr = data_new.loc[:, ['marketing_area', 'tier', 'customer_type']]
    # print(data_lr)
    train_cats(data_lr)
    # print('6666666')
    # print(data_lr)
    data_lr['c_type' + '_na'] = [1 if v == 'NA' else 0 for v in data_lr['customer_type']]
    for col in data_lr.columns:
        if not is_numeric_dtype(data_lr[col]):
            data_lr[col] = data_lr[col].cat.codes + 1
        # print(dict(enumerate(data_lr[col])))
    # print(data_lr)
    # 输出5：用户属性模块：训练集X
    X_tr_lr = data_lr.iloc[tr_idx, :]
    # print(X_tr_lr)
    # print(type(X_tr_lr))
    # 输出6：用户属性模块：测试集X
    X_te_lr = data_lr.iloc[te_idx, :]#iloc根据行号选
    # print(X_te_lr)
    # 利用 MinMaxScaler 放缩数据
    scaler = MinMaxScaler()
    scaler.fit(X_tr_lr)
    X_tr_lr[['marketing_area', 'tier', 'customer_type']] = scaler.fit_transform(
        X_tr_lr[['marketing_area', 'tier', 'customer_type']])
    X_te_lr.loc[:, ['marketing_area', 'tier', 'customer_type']] = scaler.transform(
        X_te_lr[['marketing_area', 'tier', 'customer_type']])
    # 输出13：用户属性模块的哪些字段
    categorical_vars = data_lr.columns[0:3]
    # print(categorical_vars)
    # print(X_tr_lr)
    # rint(X_te_lr)
    # print(paths)
    # print(type(paths))
    return [time_decay_tr, time_decay_te, X_tr, X_te, X_tr_lr, X_te_lr, Y_train, Y_test,
            all_X, time_decay, newlines, y, categorical_vars, paths]

# data = pd.read_excel('fake_data.xlsx')
# data['last_time_lapse'] = data['last_time_lapse'].map(eval)

if __name__ == '__main__':
    data = pd.read_excel('fake_data1.xlsx')
    seq_length = 20
    # 预处理后的数据
    data_all = process_data(data, seq_length=seq_length)
    print(data_all[-1])
    channels = list(set([_p for p in data_all[-1] for _p in p]))
    # channels.sort()
    print(channels)











