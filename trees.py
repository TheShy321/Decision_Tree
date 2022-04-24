#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Decision_Tree 
@File    ：trees.py
@IDE     ：PyCharm 
@Author  ：YuYang_Sun
@Date    ：2022-4-20 17:47 
@Introduction: 关于tree树的构建，其中包括计算不同的评价标准！
'''
from math import log
import operator
import pickle

'''
计算数据集香农熵
dataSet - 待划分的数据集
'''


def calcShannonEnt(dataSet):
    '''计算数据集的熵'''
    numEntries = len(dataSet)  # 获取数据集样本个数
    labelCounts = {}  # 初始化一个字典用来保存每个标签出现的次数
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 逐个获取标签信息
        # 如果标签没有放入统计次数字典的话，就添加进去
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0  # 初始化香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries  # 选择该标签的概率
        shannonEnt -= prob * log(prob, 2)  # 公式计算
    return shannonEnt


'''
定义按照某个特征进行划分的函数splitDataSet
输入三个变量（待划分的数据集，特征，分类值）
axis表示划分数据集的特征、value分类值
我们将对每个特征划分数据集的结果计算一次信息熵，
然后判断按照哪个特征划分数据集是最好的划分方式，
下面我们先定义一个函数，用来实现按照给定的特征划分数据集这一功能
'''

def splitDataSet(dataSet, axis, value):
    '''按照给定特征划分数据集'''
    retDataSet = []  # 创建新列表以存放满足要求的样本
    for featVec in dataSet:
        if featVec[axis] == value:
            # 下面这两句用来将axis特征去掉，并将符合条件的添加到返回的数据集中
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


'''--------------------------------------------信息增益的计算方式------------------------------------------'''

'''
#选择信息增益最大的（最优）特征作为数据集划分方式
dataSet - 待划分的数据集 ID3
'''

def chooseBeatFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 创建唯一的分类标签列表
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0.0
        # 计算每种划分方式的信息熵
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 计算最好的信息增益
        infoGain = baseEntropy - newEntropy
        # 打印每个特征的信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    print("选择%d" % bestFeature)
    # 返回信息增益最大的特征的索引值
    return bestFeature


'''--------------------------------------------信息增益率计算方式------------------------------------------'''
'''
信息增益率 需要先算出信息增益 然后在计算！
信息增益率 --> C4.5
'''


def chooseBestFeatureToSplit(dataSet):
    # 特征数量
    numFeatures = len(dataSet[0]) - 1  # 最后一列为label
    # 计算数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 信息增益
    bestInfoGain = 0.0
    # 最优特征的索引值
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征存在featList这个列表中（列表生成式）
        featList = [example[i] for example in dataSet]
        # 创建set集合{}，元素不可重复，重复的元素均被删掉
        # 从列表中创建集合是python语言得到列表中唯一元素值得最快方法
        uniqueVals = set(featList)
        # 经验条件熵
        newEntropy = 0.0
        # 计算信息增益
        for value in uniqueVals:
            # subDataSet划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益
        infoGain = baseEntropy - newEntropy
        # 打印每个特征的信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        # 计算信息增益
        if (infoGain > bestInfoGain):
            # 更新信息增益，找到最大的信息增益
            bestInfoGain = infoGain
            # 记录信息增益最大的特征的索引值
            bestFeature = i
    # 返回信息增益最大的特征的索引值
    return bestFeature


def chooseBestFeatureToSplitRatio(dataSet):
    # 特征数量
    numFeatures = len(dataSet[0]) - 1  # 最后一列为label
    # 计算数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 信息增益
    bestInfoGain = 0.0
    # 最优特征的索引值
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征存在featList这个列表中（列表生成式）
        featList = [example[i] for example in dataSet]
        # 创建set集合{}，元素不可重复，重复的元素均被删掉
        # 从列表中创建集合是python语言得到列表中唯一元素值得最快方法
        uniqueVals = set(featList)
        # 经验条件熵
        newEntropy = 0.0

        IV = 0.0
        # 计算信息增益
        for value in uniqueVals:
            # subDataSet划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            # 根据公式计算经验条件熵
            newEntropy += prob * calcShannonEnt(subDataSet)
            IV -= prob * log(prob, 2)
        # 信息增益
        infoGain = baseEntropy - newEntropy
        # 信息增益率
        if IV == 0:
            IV = 1
        infoGainRatio = infoGain / IV
        # 打印每个特征的信息增益
        print("第%d个特征的增益率为%.3f" % (i, infoGainRatio))
        # 计算信息增益率
        if (infoGainRatio > bestInfoGain):
            # 更新信息增益率，找到最大的信息增益率
            bestInfoGain = infoGainRatio
            # 记录信息增益率最大的特征的索引值
            bestFeature = i
    print("选择第%d" % bestFeature, "个特征")
    # 返回信息增益率最大的特征的索引值
    return bestFeature


'''--------------------------------------------基尼指数------------------------------------------'''
'''
基尼指数
'''


def gini(dataSet):
    # 返回数据集的行数
    numEntires = len(dataSet)
    # 保存每个标签（Label）出现次数的“字典”
    labelCounts = {}
    # 对每组特征向量进行统计
    for featVec in dataSet:
        # 提取标签（Label）信息
        currentLabel = featVec[-1]
        # 如果标签（Label）没有放入统计次数的字典，添加进去
        if currentLabel not in labelCounts.keys():
            # 创建一个新的键值对，键为currentLabel值为0
            labelCounts[currentLabel] = 0
        # Label计数
        labelCounts[currentLabel] += 1
    # 经验熵（香农熵）
    shannonEnt = 0.0
    # 计算香农熵
    for key in labelCounts:
        # 选择该标签（Label）的概率
        prob = float(labelCounts[key]) / numEntires
        # 利用公式计算
        shannonEnt = pow(prob, 2)
    # 返回经验熵（香农熵）
    print("SHANNO = ", 1 - shannonEnt)
    return 1 - shannonEnt


def chooseBestFeatureToSplitGini(dataSet):
    # 特征数量
    numFeatures = len(dataSet[0]) - 1  # 最后一列为label
    print(numFeatures)
    # 计算数据集的香农熵
    baseEntropy = calcShannonEnt(dataSet)
    # 信息增益
    bestInfoGain = 100
    # 最优特征的索引值
    bestFeature = -1
    # 遍历所有特征
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征存在featList这个列表中（列表生成式）
        featList = [example[i] for example in dataSet]
        print(featList)
        # 创建set集合{}，元素不可重复，重复的元素均被删掉
        # 从列表中创建集合是python语言得到列表中唯一元素值得最快方法
        uniqueVals = set(featList)
        # 经验条件熵
        print(uniqueVals)
        gini_index = 0.0
        # 计算信息增益
        for value in uniqueVals:
            # subDataSet划分后的子集
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算子集的概率
            prob = len(subDataSet) / float(len(dataSet))
            # print("第%%d个特征的prob为%%.3f" %% (value, prob))
            # 根据公式计算基尼指数
            gini_index += prob * gini(subDataSet)

        # 打印每个特征的信息增益
        print("第%d个特征的基尼指数为%.3f" % (i, gini_index))
        # 计算基尼指数
        if (gini_index < bestInfoGain):
            # 更新基尼指数，找到最大的基尼指数
            bestInfoGain = gini_index
            # 记录基尼指数最小的特征的索引值
            bestFeature = i
    print("选择第%d" % bestFeature, "个特征")
    # 返回信息增益最大的特征的索引值
    # 返回基尼指数最小的特征的索引值
    return bestFeature


'''--------------------------------------------构建决策树部分------------------------------------------'''

'''构建决策树'''

def majorityCnt(classList):
    '''计算出现最多的类标签 '''
    classCount = {}
    for vote in classCount:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
        sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


'''选取最优划分特征！可以通过选择信息增益、信息增益率、Gini指数来做'''


def createTree(dataSet, labels):
    '''建树'''
    classList = [example[-1] for example in dataSet]  # 获取类别标签
    if classList.count(classList[0]) == len(classList):
        return classList[0]  # 类别完全相同则停止继续划分
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)  # 遍历完所有特征时返回出现次数最多的类别
    beatFeat = chooseBeatFeatureToSplit(dataSet)  # 选取最优划分特征 -- 信息增益
    # beatFeat = chooseBestFeatureToSplitRatio(dataSet)             -- 信息增益率
    # beatFeat = chooseBestFeatureToSplitGini(dataSet)              -- 基尼系数
    beatFeatLable = labels[beatFeat]  # 获取最优划分特征对应的属性标签
    myTree = {beatFeatLable: {}}  # 存储树的所有信息
    del (labels[beatFeat])  # 删除已经使用过的属性标签
    featValues = [example[beatFeat] for example in dataSet]  # 得到训练集中所有最优特征的属性值
    uniqueVals = set(featValues)  # 去掉重复的属性值
    for value in uniqueVals:  # 遍历特征，创建决策树
        subLabels = labels[:]  # 剩余的属性标签列表
        # 递归函数实现决策树的构建
        myTree[beatFeatLable][value] = createTree(splitDataSet
                                                  (dataSet, beatFeat, value), subLabels)
    return myTree


'''决策树策树分类器'''
def classify(inputTree, featLabels, testVec):
    '''使用决策树的分类函数'''
    global classLabel
    firstStr = list(inputTree.keys())[0]  # 获取根节点
    secondDict = inputTree[firstStr]  # 获取下一级分支
    featIndex = featLabels.index(firstStr)  # 查找当前列表中第一个匹配firstStr变量的元素的索引
    # key = testVec[featIndex]  # 获取测试样本中，与根节点特征对应的取值
    # valueOfFeat = secondDict[key]  # 获取测试样本通过第一个特征分类器后的输出
    # if isinstance(valueOfFeat, dict):  # 判断节点是否为字典来以此判断是否为叶节点
    #     classLabel = classify(valueOfFeat, featLabels, testVec)
    # else:
    #     classLabel = valueOfFeat  # 如果到达叶子节点，则返回当前节点的分类标签
    # return classLabel
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    '''存储决策树'''
    fw = open(filename, 'wb')
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    '''加载决策树'''
    fr = open(filename, 'rb')
    return pickle.load(fr)


'''
用于文件读取和文本预处理数据集中的训练集和测试集
'''

def loadTrainData():
    fr = open('car/cardata.txt')
    lines = fr.readlines()
    retData = []
    for line in lines:
        items = line.strip().split(',')
        retData.append([items[i] for i in range(0, len(items))])
    return retData

def loadTestData():
    fr = open('car/car_test.txt')
    lines = fr.readlines()
    retData = []
    for line in lines:
        items = line.strip().split(',')
        retData.append([items[i] for i in range(0, len(items))])
    return retData