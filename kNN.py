#代码转自《机器学习实战》，下面是我对这段代码的注释，理解
from numpy import*
import operator
from os import listdir

#导入数据
def creatDataSet():#定义函数
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels
    
#核心代码，实现分类
def classify0(inX, dataSet, labels, k): #四个参数变量，inX表示用于分类的输入向量；dataSet训练数据集；labels标签数据集；k表示最近邻居的数据
    dataSetSize = dataSet.shape[0] #获取dataSet的行数
    diffMat = tile(inX, (dataSetSize,1)) - dataSet #将输入向量inX在列方向上增大dataSetSize倍，行方向上默认不变，即获得一个和dataSet相同大小的矩阵，然后减去dataSet
    sqDiffMat = diffMat**2 #对diffMat中每位元素做平方运算
    sqDistances = sqDiffMat.sum(axis=1) #将sqDiffMat的每一行进行相加，变成一个列向量
    distances = sqDistances**0.5 #对sqDistance的元素开方
    sortedDistIndicies = distances.argsort()#返回distance列中元素从小到大排序的索引值
    classCount={} #定义一个字典变量
    for i in range(k): #循环执行下面语句K次
        voteIlabel = labels[sortedDistIndicies[i]] #从小到大一次获取相应的labels
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #存入labels和相应键值为0，之后更新此labels的键值
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)#使用classCount的value值进行从大到小排序
    return sortedClassCount[0][0] #返回分类结果

def file2matrix(filename):#处理文件数据常用手段
    fr = open(filename)
    numberOfLines = len(fr.readlines())     
    returnMat = zeros((numberOfLines,3))        
    classLabelVector = []                       
    fr = open(filename)
    index = 0
    for line in fr.readlines(): #按行处理文件
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3] #将数据列抽取出
        classLabelVector.append(int(listFromLine[-1])) #将标签列抽取出
        index += 1
    return returnMat,classLabelVector #返回整理后的数据矩阵和标签向量

def autoNorm(dataSet):#归一化数据集，使得数据集的特征占比规整
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]#得到daaSet的行数
    normDataSet = dataSet - tile(minVals,(m,1))#扩展minVals，使它变成和daaset一样的行度
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals
