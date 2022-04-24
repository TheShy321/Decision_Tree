#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Decision_Tree 
@File    ：dataSplit.py
@IDE     ：PyCharm 
@Author  ：YuYang_Sun
@Date    ：2022-4-24 20:01
@Introduction: 关于数据分割的不同的方法
'''
import trees
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pandas as pd
data = trees.loadTrainData()
# print('data = ', data)
'''-------------------------------留出法分割数据集--------------------------------------'''

train_X , test_X = train_test_split(data, test_size=0.2, random_state=0)
''' 
X为原始数据的自变量，Y为原始数据因变量；
train_X，test_X是将X按照8:2划分所得；
train_Y，test_Y是将X按照8:2划分所得；
test_size是划分比例；
random_state设置是否使用随机数
'''


'''-------------------------------交叉验证法分割数据集--------------------------------------'''

kf = KFold(n_splits = 4, shuffle = False, random_state = None)

'''n_splits表示将数据分成几份；shuffle和random_state表示是否随机生成。
如果shuffle = False,random_state = None,重复运行将产生同样的结果；
如果shuffle = True,random_state = None,重复运行将产生不同的结果；
如果shuffle = True,random_state = （一个数值）,重复运行将产生相同的结果；
'''

data = pd.read_csv('car/cardata.csv')

for train, test in kf.split(data):
    print("%s %s" % (train, test))


'''-------------------------------交叉验证法分割数据集--------------------------------------'''

# import pandas as pd
#
# df = pd.read_csv("car/cardata.txt",delimiter=",")
#
# df.to_csv("car/cardata.csv", encoding='utf-8', index=False)