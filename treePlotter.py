'''使用文本注解绘制树节点'''

import matplotlib.pyplot as plt
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']

# 定义文本框和箭头格式
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plotNode(nodeText, centerPt, parentPt, nodeType):
    # nodeTxt为要显示的文本，centerPt为文本的中心点，parentPt为指向文本的点

    createPlot.ax1.annotate(nodeText, xytext=centerPt, textcoords="axes fraction",
                            xy=parentPt, xycoords="axes fraction",
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)

# 求叶子节点数
def getNumLeafs(myTree):
    numNode = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numNode += getNumLeafs(secondDict[key])
        else:
            numNode += 1
    return numNode


# 获取决策树的深度
def getTreeDepth(myTree):
    global thisDepth
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
    return thisDepth


# 预定义的树，用来测试
def retrieveTree(i):
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
    ]
    return listOfTrees[i]


# 绘制中间文本（在父子节点间填充文本信息）
def plotMidText(cntrPt, parentPt, txtString):
    # 求中间点的横坐标
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    # 求中间点的纵坐标
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    # 绘制树节点
    createPlot.ax1.text(xMid, yMid, txtString, va='center', ha='center', rotation=30)


# 绘制决策树
def plotTree(myTree, parentPt, nodeTxt):
    # 获得决策树的叶子节点数与深度
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    # firstStr = myTree.keys()[0]
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalw, plotTree.yOff)
    # print('c:',cntrPt)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD
    # print('d:',plotTree.yOff)
    for key in secondDict.keys():
        # 如果secondDict[key]是一颗子决策树，即字典
        if type(secondDict[key]) is dict:
            # 递归地绘制决策树
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalw
            # print('e:',plotTree.xOff)
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD
    # print('f:',plotTree.yOff)


# 创建决策树
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalw = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5 / plotTree.totalw
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.savefig('tree_ratio.png', bbox_inches='tight')  # plt.show()
