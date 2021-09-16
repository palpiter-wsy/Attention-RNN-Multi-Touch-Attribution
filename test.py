import pandas as pd
import numpy as np
# import tensorflow as tf
import matplotlib.pyplot as plt
# from sklearn.metrics import roc_curve,confusion_matrix,f1_score, roc_auc_score
# from keras.layers import Concatenate, Dot, Input, LSTM
# from keras.layers import RepeatVector, Dense, Activation
# from keras.layers import Reshape, Dropout, Add, Subtract, Flatten, Embedding
# #from keras.optimizers import Adam
#
# from keras.models import load_model, Model
# import keras.backend as K
# import warnings
#
# import process_data
# # from h5py import Dataset, Group, File
# import h5py
# warnings.filterwarnings("ignore")
#
# from process_data import *



#import feather
#%matplotlib inline

# t1 = K.variable(np.array([[[1, 2], [2, 3]], [[4, 4], [5, 3]]]))
# t2 = K.variable(np.array([[[7, 4], [8, 4]], [[2, 10], [15, 11]]]))
# d0 = K.concatenate([t1, t2], axis=-2)#axis=1
# d1 = K.concatenate([t1, t2], axis=1)
# d2 = K.concatenate([t1, t2], axis=-1)
# d3 = K.concatenate([t1, t2], axis=2)
# print(np.array([[[  1, 2]
#  , [  2,  3]
#  , [  7,  4]
#  , [  8,  4]]
# ,
#  [[ 4, 4.]
#  , [ 5, 3]
#   ,[ 2, 10]
#   ,[ 15, 11]]]).shape)
# as first layer in a sequential model:
# as first layer in a sequential model:


# with File('FFDNA_full.h5','r') as f:
# 	for k in f.keys():
# 		if isinstance(f[k], Dataset):
# 			print(f[k].value)
# 		else:
# 			print(f[k].name)

#打开.h5文件
# with h5py.File('FFDNA_full.h5',"r") as f:
#     # for key in f.keys():
#     # 	 #print(f[key], key, f[key].name, f[key].value) # 因为这里有group对象它是没有value属性的,故会异常。另外字符串读出来是字节流，需要解码成字符串。
#     #     print(f[key], key, f[key].name)
#
#     dense_group = f["dense/dense"]
#     for key in dense_group.keys():
#         print(dense_group[key], dense_group[key].name)

# array=[['A1', 'A2', 'A3', 'A4'],['A1', 'A2', 'A3'],['A1', 'A2', 'A3', 'A4'],['A1', 'A2', 'A3'],['A1', 'A2', 'A3', 'A4'],['A1', 'A2', 'A3', 'A4'],['A1', 'A2', 'A3', 'A4'],['A1', 'A2', 'A3'],['A1', 'A2', 'A3', 'A4'],['A1', 'A2', 'A3', 'A4'],['A1', 'A2', 'A3', 'A4'],['A1', 'A2', 'A3'],['A1', 'A2', 'A3'],['A1', 'A2', 'A3'],['A1', 'A2', 'A3']]
# paths=pd.Series(data=array)
#
#
# prob=np.array([[0.2914281 ],[0.69909984],[0.6896166 ] ,[0.5297201 ],[0.982928  ],[0.7747966 ],[0.8576279 ] ,[0.9235128 ] ,[0.69909984] ,[0.5297201 ] ,[0.915009  ] ,[0.52951735],[0.5185748 ] ,[0.99399245], [0.8594755 ]])
# print(prob)
# cp_idx = sorted(range(len(prob)), key=lambda k: prob[k], reverse=True)  # 从大到小排列
# print(cp_idx)
# print(cp_idx[0:3])
# cp_p = [paths[p] for p in cp_idx[0:100]]
# print(cp_p)
# cp_p_2 = set(map(tuple, cp_p))
# print(cp_p_2)
# print(list(map(list, cp_p_2)))


# print(list(map(lambda x,y:x**3,[2,3,6,8,9],[4,5,4,3])))

# chan_used=np.array([[1,2,3],[4,5,6]])
# print(chan_used)
# print(chan_used.shape[0])
# print(chan_used.shape[1])

#测试chan_used
# att_f = {m: 0 for m in range(1, 4 + 1)}
# print(att_f)
# att_count_f = {m: 0 for m in range(1, 4 + 1)}
#
# chan_used =np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4],\
#                     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3],
#                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,2,3,4]])
# r =np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],\
#                     [0,0,0,0,5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,7,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,8,0,0,0,0,0,0,0,0,0,0,0,0],
#                      [0,0,0,0,0,0,0,0,9,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,10,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,11,0,0,0,0,0,0,0,0,0]])
# channels=['A2', 'A4', 'A1', 'A3']
# # print('chan_used:\n')
# # print(chan_used)
# for m in range(chan_used.shape[0]):
#     for n in range(chan_used.shape[1]):
#         if chan_used[m, n] != 0:
#             att_f[chan_used[m, n]] += r[m, n]
#             att_count_f[chan_used[m, n]] += 1
# print(att_f[1])
# print(chan_used[0,16])
# print(chan_used[1,16])
# for n in range(4):
#     att_f[channels[n]] = att_f.pop(n + 1)
# print('att_f    att_count_f:\n')
# print(att_f)
# print(att_count_f)

#测试plot函数
# plt.plot([0, 1], [0, 1], 'k--')
# plt.axis([0, 1, 0, 1])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.show()

#测试混淆矩阵
# y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
# y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
# print(confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"]))

# def sum_demo(x, y):
#     for _ in range(2):
#         x += 1
#         y += 1
#         result = x + y
#     return result
#
# if __name__ == '__main__':
#     result = sum_demo(1, 1)
#     print(result)


L=[['EMAIL', 'YOUTUBE', 'PROGRAMMATIC'], ['EMAIL', 'SOCIAL', 'SEM'], ['EMAIL', 'YOUTUBE', 'SOCIAL', 'EMAIL', 'PROGRAMMATIC'], ['SOCIAL', 'EMAIL', 'EMAIL', 'EMAIL', 'EMAIL', 'SEM'], ['EMAIL', 'SOCIAL', 'SOCIAL', 'EMAIL'], ['EMAIL', 'SOCIAL', 'PROGRAMMATIC'], ['EMAIL', 'YOUTUBE', 'SOCIAL'], ['SOCIAL', 'SEM', 'SOCIAL', 'EMAIL', 'PROGRAMMATIC', 'SEM'], ['EMAIL', 'SOCIAL', 'SOCIAL'], ['EMAIL', 'YOUTUBE', 'SOCIAL', 'SOCIAL', 'EMAIL', 'PROGRAMMATIC'], ['SOCIAL', 'YOUTUBE', 'EMAIL', 'PROGRAMMATIC'], ['EMAIL', 'YOUTUBE', 'EMAIL', 'EMAIL'], ['EMAIL', 'EMAIL', 'SEM'], ['EMAIL', 'SOCIAL', 'EMAIL', 'SOCIAL', 'SEM'], ['PROGRAMMATIC', 'YOUTUBE', 'SOCIAL'], ['EMAIL', 'YOUTUBE', 'SOCIAL', 'EMAIL', 'PROGRAMMATIC', 'SEM'], ['EMAIL', 'EMAIL', 'PROGRAMMATIC'], ['SOCIAL', 'EMAIL', 'EMAIL'], ['EMAIL', 'YOUTUBE', 'EMAIL', 'SOCIAL', 'SOCIAL'], ['EMAIL', 'SOCIAL', 'EMAIL', 'EMAIL', 'SEM'], ['PROGRAMMATIC', 'PROGRAMMATIC', 'EMAIL'], ['EMAIL', 'PROGRAMMATIC', 'SEM'], ['EMAIL', 'PROGRAMMATIC', 'PROGRAMMATIC'], ['EMAIL', 'SOCIAL', 'EMAIL'], ['SOCIAL', 'SEM', 'SOCIAL', 'SEM'], ['SOCIAL', 'YOUTUBE', 'PROGRAMMATIC'], ['EMAIL', 'YOUTUBE', 'EMAIL', 'EMAIL', 'EMAIL', 'SOCIAL'], ['SOCIAL', 'SOCIAL', 'EMAIL', 'PROGRAMMATIC'], ['EMAIL', 'YOUTUBE', 'PROGRAMMATIC', 'EMAIL', 'PROGRAMMATIC', 'SEM'], ['EMAIL', 'EMAIL', 'SOCIAL', 'EMAIL', 'PROGRAMMATIC'], ['SOCIAL', 'SEM', 'EMAIL', 'PROGRAMMATIC'], ['EMAIL', 'YOUTUBE', 'EMAIL', 'PROGRAMMATIC', 'SEM'], ['SOCIAL', 'EMAIL', 'EMAIL', 'PROGRAMMATIC'], ['EMAIL', 'EMAIL', 'EMAIL'], ['SOCIAL', 'EMAIL', 'SOCIAL', 'EMAIL', 'PROGRAMMATIC', 'SEM'], ['SOCIAL', 'EMAIL', 'PROGRAMMATIC'], ['EMAIL', 'YOUTUBE', 'PROGRAMMATIC', 'PROGRAMMATIC', 'EMAIL', 'PROGRAMMATIC'], ['SOCIAL', 'SEM', 'SOCIAL', 'SOCIAL', 'SOCIAL', 'SEM', 'SEM'], ['SOCIAL', 'EMAIL', 'SOCIAL']]
A=np.array(L)
print(type(L))
print(A.shape)