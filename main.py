from numpy import *
from sklearn.datasets import load_iris
from sklearn import datasets
import trees
import treePlotter

from sklearn.model_selection import train_test_split

data = trees.loadTrainData()
carData , TestcarData = train_test_split(data, test_size=0.2, random_state=0)
# carData = trees.loadTrainData()
# TestcarData = trees.loadTestData()

carLabels = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
#          「买入价」，「维护费」，「车门数」，「可容纳人数」，「后备箱大小」，「安全性」汽车测评的数据集

carTree = trees.createTree(carData, carLabels)
treePlotter.createPlot(carTree)
# trees.createPlot(carTree)
ans = [example[-1] for example in TestcarData]
result = []

# 测试数据
for i in range(len(TestcarData)):
    resu = trees.classify(carTree, ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'], TestcarData[i])

    result.append(resu)  # 在列表末尾添加新的对象'

accuracy = 0.0
for i in range(len(TestcarData)):
    result = array(result)  # 从队列中取出
    ans = array(ans)
    accuracy = mean(result == ans)

print("训练数据", len(carData), "份,测试数据", len(TestcarData), "份,准确率为:", accuracy)