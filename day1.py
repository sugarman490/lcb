from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import jieba  # 分割中文句子


# 字典数据抽取：把字典中一些类别数据分别转换成特征

def dictvec():
    """字典数据抽取"""
    dict = DictVectorizer(sparse=False)
    data = dict.fit_transform([{'city': '北京', 'temperature': 100}, {'city': '上海', 'temperature': 60},
                               {'city': '深圳', 'temperature': 30}, {'city': '苏州', 'temperature': 10},
                               {'city': '深圳', 'temperature': 20}])
    print(dict.get_feature_names_out())
    # 把类别型特征转化为one-hot编码，利于分析
    print(dict.inverse_transform(data))
    print(data)
    return None


def countvec():
    """对文本进行特征值化"""
    cv = CountVectorizer()
    data = cv.fit_transform(["Life life is short,i like python", "Life is too long,i dislike python"])
    print(cv.get_feature_names_out())
    print(data.toarray())  # 对每篇文章，在词的列表里面进行统计每个词出现的次数(对单个字母不统计)
    return None


def countvecCN():
    """对文本进行特征值化"""
    cv = CountVectorizer()
    data = cv.fit_transform(["人生 苦短,我 喜欢 python", "人生 太长,我 不喜欢 python"])
    print(cv.get_feature_names_out())
    print(data.toarray())  # 对每篇文章，在词的列表里面进行统计以空格分隔的每个词出现的次数(对单个字不统计)
    return None


def cutword():
    con1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    con2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。")
    con3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")
    # 转换成列表
    content1 = list(con1)
    content2 = list(con2)
    content3 = list(con3)
    # 把列表转化成字符串
    c1 = ' '.join(content1)
    c2 = ' '.join(content2)
    c3 = ' '.join(content3)
    return c1, c2, c3


def hanzivec():
    """中文特征值化"""
    c1, c2, c3 = cutword()
    print(c1, c2, c3)
    cv = CountVectorizer()
    data = cv.fit_transform([c1, c2, c3])
    print(cv.get_feature_names_out())
    print(data.toarray())  # 对每篇文章，在词的列表里面进行统计以空格分隔的每个词出现的次数(对单个字不统计)
    return None


def tfidfvec():
    """分类文章"""
    print("tfidf")
    c1, c2, c3 = cutword()
    print(c1, c2, c3)
    tf = TfidfVectorizer()
    data = tf.fit_transform([c1, c2, c3])
    print(tf.get_feature_names_out())
    print(data.toarray())  # 对每篇文章，在词的列表里面进行统计以空格分隔的每个词出现的次数(对单个字不统计)
    return None


def mm():
    """归一化处理(几个特征重要程度相同)"""
    print("归一化处理")
    mm = MinMaxScaler(feature_range=(2, 3))
    data = mm.fit_transform([[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]])
    print(data)
    return None


def stand():
    """标准化缩放"""
    print("标准化缩放")
    std = StandardScaler()
    data = std.fit_transform([[1, -1, 3], [2, 4, 2], [4, 6, -1]])
    print(data)
    return None


def im():
    """缺失值处理"""
    print("缺失值处理")
    im = SimpleImputer(missing_values=np.nan, strategy='mean')
    data = im.fit_transform([[1, 2], [np.nan, 3], [7, 6]])
    print(data)
    return None


def var():
    """特征选择-删除低方差特征"""
    print("删除低方差特征")
    var = VarianceThreshold(threshold=0.0)  # 删除方差为0的特征
    data = var.fit_transform([[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]])
    print(data)
    return None


def pca():
    # 主成分分析进行数据降维
    print("主成分分析进行数据降维")
    pca = PCA(n_components=0.9)
    # n_components为整数时，代表特征维数，为小数（如0.9）时，代表原数据量的90%
    data = pca.fit_transform([[2, 8, 4, 5], [6, 3, 0, 8], [5, 4, 9, 1]])
    print(data)
    return None


if __name__ == '__main__':
    dictvec()
    countvec()
    countvecCN()
    hanzivec()
    tfidfvec()
    mm()
    stand()
    im()
    var()
    pca()
