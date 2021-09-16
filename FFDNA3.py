import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, f1_score, roc_auc_score
from keras.layers import Concatenate, Dot, Input, LSTM
from keras.layers import RepeatVector, Dense, Activation
from keras.layers import Reshape, Dropout, Add, Subtract, Flatten, Embedding
# from keras.optimizers import Adam

from keras.models import load_model, Model
import keras.backend as K
import warnings

import process_data

warnings.filterwarnings("ignore")

from process_data3 import *


# import feather
# %matplotlib inline
# 定义一个FFDNA类
class FFDNA(object):

    def __init__(self, data, config):
        self.time_decay_tr = data[0]  # 训练集时间
        self.time_decay_te = data[1]  # 测试集时间
        self.X_tr = data[2]  # 用户路径模块-训练集X
        self.X_te = data[3]  # 用户路径模块-测试集X
        self.X_tr_lr = data[4]  # 用户属性模块-训练集X
        self.X_te_lr = data[5]  # 用户属性模块-测试集X
        self.Y_train = data[6]
        self.Y_test = data[7]
        self.all_X = data[8]  # 所有x
        self.time_decay = data[9]  # 所有时间
        self.newlines = data[10]
        self.y = data[11]  # 所有y
        self.categorical_vars = data[12]  # 用户属性模块 哪些字段
        self.paths = data[13]  # 所有路径
        self.config = config
        self.channels = config['channels']
        self.Tx = config['Tx']
        self.n_a = config['n_a']
        self.n_s = config['n_s']
        self.s0 = config['s0']
        self.s1 = config['s1']
        self.vocab_size = config['vocab_size']
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.learning_rate = config['learning_rate']

    def save_weight(self, name, model):
        model.save_weights(name)

    def load_weight(self, name):
        self.model.load_weights(name)

    '''
    attention层：如何计算三个输入：
    先计算attention weights：
    将三个Input，concatenator([s_prev,a])，然后求差Subtract(name='data-time')([energies,t0])，
    进入activator激活层中特别的softmax，计算attention weights
    然后将attention weights 点积 路径的lstm隐含层，得到输出项

    这里的attention weights就是非常关键的每个节点的权重了
    '''

    def one_step_attention(self, a, s_prev, t0):
        # repeator = RepeatVector(Tx)#不改变步长，改变每一步的维数（属性长度）
        repeator = RepeatVector(Tx)  # 不改变步长，改变每一步的维数（属性长度），RepeatVector层将输入重复n次
        concatenator = Concatenate(axis=-1)  # 从倒数第一个维度进行拼接
        densor1 = Dense(10, activation="tanh")  # 定义10个神经元节点，使用tanh激活函数的神经层
        densor2 = Dense(1, activation="relu")
        # softmax函数：有限离散概率分布的梯度对数归一化
        activator = Activation(self.softmax,name='attention_weights')  # We are using a custom softmax(axis = 1) loaded in this notebook
        dotor = Dot(axes=1)
        # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a".
        s_prev = repeator(s_prev)
        # Use concatenator to concatenate a and s_prev on the last axis
        concat = concatenator([s_prev, a])  # so(m,Tx,n_s),a=n_a=32
        # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e.
        e = densor1(concat)
        # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies.
        energies = densor2(e)
        # Use "activator" on "energies" to compute the attention weights "alphas"
        energies = Subtract(name='data-time')([energies, t0])
        alphas = activator(energies)
        # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next layer
        context = dotor([alphas, a])
        # print('1')
        # print(concat.shape)
        # print('2')
        # print(e.shape)
        # print('3')
        # print(energies.shape)
        return context

    '''
    customer profile---embedding layer+ANN:额外融入用户属性信息
    一个简单的全连接神经网络来处理客户数据。这部分非常简单，只有几个密集的层。
    之前用户编码会用one-hot encoding，这里使用的是embedding layer自训练。

    嵌入层 Embedding:将正整数（索引值）转换为固定尺寸的稠密向量。
    例如： [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
    下面是用户属性模块的embedding
    这里每个用户属性categorical_vars，都会进行一次Embedding，然后之后拼接在一起Concatenate()(embeddings)，进入一个dense(10)全连接层。
    该模块的输入：inputss，需要同时给入三个属性变量，
    输出：out_control 32维的全连接输出
    '''

    def build_embedding_network(self, no_of_unique_cat=83, output_shape=32):
        inputss = []
        embeddings = []
        for c in self.categorical_vars:
            inputs = Input(shape=(1,), name='input_sparse_' + c)  # 表示输入为1维向量，字符串形式表示当前层的名字
            # no_of_unique_cat  = data_lr[categorical_var].nunique()
            embedding_size = min(np.ceil((no_of_unique_cat) / 2), 50)  # 83/2向上取整=42
            embedding_size = int(embedding_size)
            embedding = Embedding(no_of_unique_cat + 1, embedding_size, input_length=1)(inputs)  # 没看懂？？？？？？？？？
            embedding = Reshape(target_shape=(embedding_size,))(embedding)
            inputss.append(inputs)
            embeddings.append(embedding)
        input_numeric = Input(shape=(1,), name='input_constinuous')
        embedding_numeric = Dense(16)(input_numeric)
        inputss.append(input_numeric)
        embeddings.append(embedding_numeric)

        x = Concatenate()(embeddings)

        x = Dense(10, activation='relu')(x)
        x = Dropout(.15)(x)  # 需要丢弃的比率，这一层神经元经过dropout后，1000个神经元中会有大约150个的值被置为0.
        out_control = Dense(output_shape)(x)
        return inputss, out_control  # out_control为32维全连接输出

    def softmax(self, x, axis=1):
        ndim = K.ndim(x)
        if ndim == 2:
            return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims=True)
            return e / s
        else:
            raise ValueError('Cannot apply softmax to a tensor that is 1D')

    def model(self):
        '''
        模型初始化

        '''
        # Define the inputs of your model with a shape (Tx,)
        # Define s0, initial hidden state for the decoder LSTM of shape (n_s,)
        '''
        RNN with Attention将时间衰减作为attention加入

        这里主要是路径模块，完全可以当做文本来看待，这里有三个需要输入的：

        input_att ，第一个输入，主要记录路径的，其中self.vocab_size一般是总词量 +1，这边就是所有路径节点数+ 1（5个）；
        self.Tx 代表文字的padding 长度，这里是20，之后 -> 进入到LSTM -> attention层
        s0 = Input(shape=(self.n_s,), name='s0')，初始化decoder LSTM隐藏层 -> attenion层
        t0 = Input(shape=(self.Tx,1), name='input_timeDecay') 时间维度因素 -> attention层
        '''

        input_att = Input(shape=(self.Tx, self.vocab_size), name='input_path')  # 预期输入的是20*5
        s0 = Input(shape=(self.n_s,), name='s0')  # n_s=64
        s = s0
        # input time decay data
        t0 = Input(shape=(self.Tx, 1), name='input_timeDecay')  # 输入：20*1
        t = t0
        # Step 1: Define pre-attention LSTM.
        a = LSTM(self.n_a, return_sequences=True)(input_att)  # 输出空间的维度：n_a=32
        # Step 2: import attention model
        context = self.one_step_attention(a, s, t)
        c = Flatten()(context)  # flatten层用来将输入“压平”，把多维输入一维化，常用在从卷积层到全连接层的过渡。
        '''
        融合层：路径模块和客户属性模块，输出到另一个dense层，然后由sigmod激活函数到最终0/1分类
        '''
        out_att = Dense(32, activation="sigmoid", name='single_output')(c)  # c为路径模块输出

        # Step 3: import embedding data for customer-ralated variables 导入客户变量相关的嵌入数据
        input_con, out_control = self.build_embedding_network()
        added = Add()([out_att, out_control])  # 将两模块的输出直接相加add()，不是conatenate()
        out_all = Dense(1, activation='sigmoid')(added)  # 输出为1维向量
        # Step 4: Create model instance taking three inputs and returning the list of outputs.
        self.model = Model([input_att, s0, t0, input_con[0],
                            input_con[1], input_con[2], input_con[3],input_con[4],input_con[5],input_con[6]], out_all)
        # print('s.shape is: ')
        # print(s.shape)
        # print('input_att.shape is :' )
        # print(input_att.shape)
        # print('out_att.shape is :')
        # print(out_att.shape)
        # print(np.array(input_con).shape)
        # print(out_control.shape)
        # print(out_all.shape)
        # self.model.

        print(self.model.summary())
        # return self.model

    '''
    训练与预测：
    从model.fit来看，这里需要Input的内容非常多；而且，这里的用户属性self.X_tr_lr.iloc[:,1],self.X_tr_lr.iloc[:,2],self.X_tr_lr.iloc[:,3]为什么分为三个？
    因为用户属性每个属性特征都需要独立embedding
    '''

    def train_model(self, save_name, loss='binary_crossentropy', opt='adam',
                    metrics=['accuracy']):  # 交叉熵损失函数，Adam优化算法；metrics=accuracy评估方法
        self.model.compile(loss=loss, optimizer=opt, metrics=metrics)
        self.history = self.model.fit(
            [self.X_tr, self.s0, self.time_decay_tr, self.X_tr_lr.iloc[:, 0], self.X_tr_lr.iloc[:, 1],self.X_tr_lr.iloc[:, 2], self.X_tr_lr.iloc[:, 3], self.X_tr_lr.iloc[:, 4], self.X_tr_lr.iloc[:, 5], self.X_tr_lr.iloc[:, 6]],self.Y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=2)  # 2为每个epoch输出一行记录

        self.save_weight(save_name, self.model)

    # model performance
    def plot_roc_curve(self, fpr, tpr, label=None):
        plt.plot(fpr, tpr, linewidth=2, label=label,color='r')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.show()

    def metric(self, y_valid, prob, cl, label=None):
        #y_true为真正二进制标签，y_score为目标得分，fpr为增加假阳性率(把负样本预测为正例的概率)，tpr为增加真阳性率(把正样本预测为正例的概率)，均为array，shape=[>2],thresholds : array, shape = [n_thresholds]
        #ROC（Receiver Operating Characteristic）曲线是以假正率（FPR）和真正率（TPR）为轴的曲线，ROC曲线下面的面积我们叫做AUC,AUC越大说明性能越好
        fpr, tpr, threshold = roc_curve(y_valid, prob)
        auc = roc_auc_score(y_valid, prob)
        self.plot_roc_curve(fpr, tpr, label=label)
        acc = (y_valid == cl).mean()
        print('Accuracy: {:.3f}, AUC: {:.3f}'.format(acc, auc))

    def plot_loss(self):
        ylims = range(1, self.epochs + 1, 10)
        plt.plot(self.history.history['loss'], color='red', label='train loss')
        plt.xticks(ylims)
        plt.legend(loc=1)
        plt.title('train loss vs epochs')

    def plot_acc(self):
        ylims = range(1, self.epochs + 1, 10)
        plt.plot(self.history.history['acc'], label='acc', c='r')
        plt.xticks(ylims)
        plt.legend(loc=4)
        plt.title('train acc vs epochs')

    def auc_score_train(self, threshold):
        prob = self.model.predict(
            [self.X_tr, self.s0, self.time_decay_tr, self.X_tr_lr.iloc[:, 0], self.X_tr_lr.iloc[:, 1],
             self.X_tr_lr.iloc[:, 2], self.X_tr_lr.iloc[:, 3], self.X_tr_lr.iloc[:, 4], self.X_tr_lr.iloc[:, 5],self.X_tr_lr.iloc[:, 6]])#输出为n维数组
        cl = [1 if p > threshold else 0 for p in prob]  #01矩阵
        print(confusion_matrix(self.Y_train, cl))#混淆矩阵，输出为n维数组
        print(self.metric(self.Y_train, prob, cl, label='train dataset performance'))

    def auc_score_test(self, threshold):
        prob = self.model.predict(
            [self.X_te, self.s1, self.time_decay_te, self.X_te_lr.iloc[:, 0], self.X_te_lr.iloc[:, 1],
             self.X_te_lr.iloc[:, 2], self.X_te_lr.iloc[:, 3], self.X_te_lr.iloc[:, 4], self.X_te_lr.iloc[:, 5],self.X_te_lr.iloc[:, 6]])
        cl = [1 if p > threshold else 0 for p in prob]
        print(confusion_matrix(self.Y_test, cl))
        print(self.metric(self.Y_test, prob, cl, label='test dataset performance'))

    def test_model(self, threshold, train=False):
        if train:
            self.auc_score_train(threshold)
        else:
            self.auc_socre_test(threshold)

    # credits for different channels; as the input data for budget calculation formula
    '''
    确定每个节点的权重：
    在原函数的attributes,由于数据是自己造的，channel不一样，所以需要自己改造一下这个函数。
    就可以得到每个节点的权重。
    '''

    def attributes(self):
        # 获得每个节点权重
        layer = self.model.layers[29]
        # print(self.model().layers[29])
        m_all, _, _ = self.all_X.shape  # 训练集m_all=69
        self.s_all = np.zeros((m_all, self.n_s))  # 69*64
        # 使用k.function提取中间层的输出，不是很懂
        # K.function()实例化一个Keras函数
        f_f = K.function([self.model.input[0], self.model.input[1], self.model.input[2]],
                         [layer.output])
        # print("$$$$$$$$$$")
        # print(layer.output)
        # print(self.model.input[0].shape, self.model.input[1].shape, self.model.input[2].shape)
        # print(self.all_X[self.y == 1].shape, self.s_all[self.y == 1].shape, self.time_decay[self.y == 1].shape)
        r = f_f([self.all_X[self.y == 1], self.s_all[self.y == 1],
                 self.time_decay[self.y == 1]])[0].reshape(self.all_X[self.y == 1].shape[0],
                self.all_X[self.y == 1].shape[1])
        # print('r:')
        # print(r)

        # att_f = {m:0 for m in range(1,6)}
        # att_count_f = {m:0 for m in range(1,6)}
        att_f = {m: 0 for m in range(1, n_channels + 1)}
        att_count_f = {m: 0 for m in range(1, n_channels + 1)}

        chan_used = self.newlines[self.y == 1]
        # print('chan_used:\n')
        # print(chan_used)
        for m in range(chan_used.shape[0]):
            for n in range(chan_used.shape[1]):
                if chan_used[m, n] != 0:
                    att_f[chan_used[m, n]] += r[m, n]
                    att_count_f[chan_used[m, n]] += 1
        # print(att_f)
        for n in range(n_channels):
            att_f[channels[n]] = att_f.pop(n + 1)
        # print('att_f    att_count_f:\n')
        # print(att_f)
        # print(att_count_f)

        return att_f

    '''
    下游应用3：确定最有影响力的路径
    根据每个客户路径的转换概率排名，列出最具影响力的N条路径
    '''

    def critical_paths(self):
        # 当使用predict()方法进行预测时，返回值是数值，表示样本属于每一个类别的概率，我们可以使用numpy.argmax()方法找到样本以最大概率所属的类别作为样本的预测标签。
        # prob = self.model.predict(
        #     [self.X_tr, self.s0, self.time_decay_tr, self.X_tr_lr.iloc[:, 0], self.X_tr_lr.iloc[:, 1],
        #      self.X_tr_lr.iloc[:, 2], self.X_tr_lr.iloc[:, 3]])
        # cp_idx = sorted(range(len(prob)), key=lambda k: prob[k], reverse=True)
        # # print([prob[p] for p in cp_idx[0:100]])
        # cp_p = [self.paths[p] for p in cp_idx[0:100]]
        #
        # cp_p_2 = set(map(tuple, cp_p))
        # print(list(map(list, cp_p_2)))

        prob = self.model.predict([self.X_tr, self.s0, self.time_decay_tr, self.X_tr_lr.iloc[:, 0], self.X_tr_lr.iloc[:, 1], self.X_tr_lr.iloc[:, 2], self.X_tr_lr.iloc[:, 3], self.X_tr_lr.iloc[:, 4], self.X_tr_lr.iloc[:, 5],self.X_tr_lr.iloc[:, 6]])
        # 训练集预测 - 找到预测概率比较高的路径
        cp_idx = sorted(range(len(prob)), key=lambda k: prob[k], reverse=True)  # 从大到小排列
        # print(cp_idx)
        cp_p = [self.paths[p] for p in cp_idx[0:100]]

        cp_p_2 = set(map(tuple, cp_p))
        print(list(map(list, cp_p_2)))
        L=np.array(list(map(list, cp_p_2)))
        # print(L.shape)


if __name__ == '__main__':
    # got data
    # data = pd.read_csv('df_paths_noblank_purchase.csv')
    data = pd.read_excel('fake_data2.xlsx')
    seq_length = 20
    # 预处理后的数据
    data_all = process_data(data, seq_length=seq_length)
    '''
    time_decay_tr, 训练集时间
    time_decay_te,测试集时间
    X_tr, 用户路径模块 -训练集X(15*20*5)   one-hot
    X_te,用户路径模块 - 测试集X(2*20*5)    one-hot
    X_tr_lr,用户属性模块-训练集X 归一化后
    X_te_lr,用户属性模块-测试集X  归一化后
    Y_train,   
    Y_test,    
    all_X, 所有x one-hot
    time_decay,所有时间
    newlines.shape,17*20,pad_sequence
    y ,所有y
    categorical_vars,用户属性模块 哪些字段
    paths , 打乱后的训练集所有路径
    '''

    # hyper parameters超参数
    n_a = 32
    n_s = 64
    m = data_all[2].shape[0]  # 训练集个数15
    m_t = data_all[3].shape[0]  # 测试集个数2
    s0 = np.zeros((m, n_s))  # 15*64
    s1 = np.zeros((m_t, n_s))  # 2*64
    batch_size = 64
    Tx = seq_length
    learning_rate = 0.001

    # channels = ['Natural Search','Email','Paid Search','Media','Social']
    channels = list(set([_p for p in data_all[-1] for _p in p]))  # ['A2', 'A4', 'A1', 'A3']列表去重

    n_channels = len(channels)  # 5
    vocab_size = n_channels + 3  # 8
    epochs = 120
    config = {'channels': channels, 'Tx': Tx, 'n_a': n_a, 'n_s': n_s, 's0': s0, 's1': s1, 'vocab_size': vocab_size,'epochs': epochs, 'batch_size': batch_size, 'learning_rate': learning_rate}

    # model
    # ana_mta_model是FFDNA类的一个具体的对象
    ana_mta_model = FFDNA(data_all, config)
    # 模型初始化
    ana_mta_model.model()

    # 模型训练
    save_name = 'FFDNA_full.h5'
    ana_mta_model.train_model(save_name, loss='binary_crossentropy', opt='adam', metrics=['accuracy'])

    # 模型重载
    ana_mta_model = FFDNA(data_all, config)
    ana_mta_model.model()
    ana_mta_model.load_weight('FFDNA_full.h5')

    # 训练集预测:找到预测概率比较高的路径
    # keras predict()函数，返回值为数值，表示样本属于每一个类别的概率，可使用numpy.argmax()方法找到样本以最大概率所属的类别作为样本的预测标签。
    # prob = ana_mta_model.model.predict([ana_mta_model.X_tr, ana_mta_model.s0, ana_mta_model.time_decay_tr, \
    #                                     ana_mta_model.X_tr_lr.iloc[:, 0], \
    #                                     ana_mta_model.X_tr_lr.iloc[:, 1],
    #                                     ana_mta_model.X_tr_lr.iloc[:, 2], ana_mta_model.X_tr_lr.iloc[:, 3]])
    # print('aaaaaaaaaaaaaaaa')
    # print(prob)
    # print(prob.shape)

    # pred需要包括:
    #     - ana_mta_model.X_tr# 用户路径特征
    #     - ana_mta_model.s0 # 15*64 训练集
    #     - ana_mta_model.time_decay_tr # 训练集时间
    #     - ana_mta_model.X_tr_lr.iloc[:,0] # 用户属性特征 - marketing_area
    #     - ana_mta_model.X_tr_lr.iloc[:,1] # 用户属性特征 - tier
    #     - ana_mta_model.X_tr_lr.iloc[:,2] # 用户属性特征 - customer_type
    #     - ana_mta_model.X_tr_lr.iloc[:,3] # 用户属性特征 - c_type_na,有但是不知道有啥用

    # 训练集预测 - 找到预测概率比较高的路径
    # cp_idx = sorted(range(len(prob)), key=lambda k: prob[k], reverse=True)  # 从大到小排列
    # # print(cp_idx)
    # cp_p = [ana_mta_model.paths[p] for p in cp_idx[0:100]]
    #
    # cp_p_2 = set(map(tuple, cp_p))
    # print(list(map(list, cp_p_2)))



    # att_f[m.channels[0]] = att_f.pop(1)
    # att_f[m.channels[1]] = att_f.pop(2)
    # att_f[m.channels[2]] = att_f.pop(3)
    # att_f[m.channels[3]] = att_f.pop(4)
    # att_f[m.channels[4]] = att_f.pop(5)

    # #m.train_model('s.h5')
    print('\n\n 1. Test dataset performance:\n')
    ana_mta_model.auc_score_test(0.5)
    print('\n\n 2. Train performance:\n')
    ana_mta_model.auc_score_train(0.5)
    att_f = ana_mta_model.attributes()
    print('\n\n 3. Channel credits: \n',att_f)
    print('\n\n 4. Top critical paths: \n')
    ana_mta_model.critical_paths()

    '''
    可以进行拓展的方面：https://mattzheng.blog.csdn.net/article/details/118469466
    确定购买潜力以及其他更多的变形
    如果有顾客点击了很多路径内容还没转化，可以通过模型得到他购买的可能性。

    例如，如果你的公司也关心每个渠道其他转化情况(如电子邮件活动中的广告的点击率)，可在LSTM之上添加更多的层来实现这一点，

    此外，还可以预测一次购买的平均支出或金额，这可能会使分配权重更准确，也可以提供您关于调整供应链的信息。可以参考上文的输出接入：average spending

    沿着这个一直再设想一下，一切NLP的模型都可以使用上；
    比如利用预训练模型，后续可以接上非常多的应用，包括预测用户下一个点击页面是什么（NSP）
    总之，这块应该还有非常多空间可以思考与继续深究
    '''








