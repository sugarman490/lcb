from sklearn.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

li = load_iris()
# print("获取特征值")
# print(li.data)
# print("目标值")
# print(li.target)
# print(li.DESCR)

# 注意返回值，训练集train  x_train(特征值) y_train(目标值)   测试集test x_test y_test
# x_train,x_test,y_train,y_test=train_test_split(li.data,li.target,test_size=0.25)
# print("训练集特征值和目标值：",x_train,y_train)
# print("测试集特征值和目标值：",x_test,y_test)

# 分类数据集
# news = fetch_20newsgroups(data_home="./data", subset='all')
# print(news.data)
# print(news.target)
lb = load_boston()
# 回归数据集
print("获取特征值")
print(lb.data)
print("目标值")
print(lb.target)
print(lb.DESCR)

s = StandardScaler()
sd = s.fit_transform([[1, 2, 3], [4, 5, 6]])
print("fit_transform", sd)

ss = StandardScaler()
ss.fit([[1, 2, 3], [4, 5, 6]])  # 计算平均值和标准差
ssd = ss.transform([[1, 2, 3], [4, 5, 6]])
print("fit+transform", ssd)


def naviebayes():
    """朴素贝叶斯进行文本分类"""
    news = fetch_20newsgroups(subset='all')
    # 数据分割
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25)
    # 对数据集进行特征抽取
    tf = TfidfVectorizer()
    # 以训练集当中的词的列表进行每篇文章重要性统计['a','b','c','d']
    x_train = tf.fit_transform(x_train)
    print(tf.get_feature_names())
    x_test = tf.transform(x_test)
    # 朴素贝叶斯算法预测
    mlt = MultinomialNB(alpha=1.0)
    print(x_train.toarray())
    mlt.fit(x_train, y_train)
    y_predict = mlt.predict(x_test)
    print("预测文章类别为：", y_predict)
    print("准确率为：", mlt.score(x_test, y_test))
    print("每个类别的精确率和召回率", classification_report(y_test, y_predict, target_names=news.target_names))
    return None


def decision():
    """决策树对泰坦尼克号进行预测生死"""
    # 获取数据
    titan = pd.read_csv("https://datahub.csail.mit.edu/download/jander/historic/file/titanic.csv")
    # 处理数据，找出特征值和目标值
    x = titan[['pclass', 'age', 'sex']]
    y = titan['survived']
    print(x)
    # 缺失值处理
    x['age'].fillna(x['age'].mean(), inplace=True)
    # 先分割数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    # 进行处理（特征工程） 特征->类别->one_hot编码
    dict = DictVectorizer(sparse=False)
    # 转成字典形式
    x_train = dict.fit_transform(x_train.to_dict(orient="records"))
    print(dict.get_feature_names())
    x_test = dict.transform(x_test.to_dict(orient="records"))
    # # print(x_train)
    # # 用决策树进行预测
    # dtc = DecisionTreeClassifier()
    # dtc.fit(x_train, y_train)
    # # 预测准确率
    # print("预测的准确率：", dtc.score(x_test, y_test))
    # #导出决策树的结构
    # export_graphviz(dtc,out_file="./tree.dot",feature_names=['年龄','a阶级','b阶级','c阶级','女','男'])

    # 随机森林（超参数调优）
    rf = RandomForestClassifier()
    param = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    # 网络搜索与交叉验证
    gc = GridSearchCV(rf, param_grid=param, cv=2)
    gc.fit(x_train, y_train)
    print("准确率：", gc.score(x_test, y_test))
    print("查看选择的参数模型", gc.best_params_)
    return None


if __name__ == '__main__':
    decision()
