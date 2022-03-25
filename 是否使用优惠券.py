# 导入python库
import pandas as pd # pandas : 数据分析库
from skLearn. modeL_ selection import train_ test_ split # sklearn: 机器学习
库
from skLearn. metrics import cLassification_ report, confusion_ matrix
from skLearn. tree import Decis ion TreeClassifier
from matplotlib import pyplot as plt # matplotlib:数据可视化库
import matplotlib as mpL

plt. rcParams[ 'font . sans-serif'] = [ 'simhei'] #中文显示
mpl. rcParams[ 'axes.unicode_ minus'] = False #负号显示
import warnings # 发出警告，或者忽略它或引发异常。

warnings.filterwarnings( ' ignore ')
import numpy as np
from skLearn import tree
import pydotpLus

#获取信用卡欺诈数据
data = pd.read_ .csv(' 优惠券数据. csv')
#数据预处理
#数据存在空值，删除空值
data = data . dropna()
#删除完全一样的数据，去重
data = data.drop_ duplicates()
#数据分析:统计欺诈非欺诈数量
num_ 1 = len(data[data[ '该次活动中是否有使用优惠券'] == 1])
num_日= len(data[ data['该次活动中是否有使用优惠券'] == 0])
print('使用优惠券的数量:，num 1)
print('没有使用优惠券的数量:，num 0)
#构造训练数据
#获取类别数据
target = data[ '该次活动中是否有使用优惠券']. values
#获取列名列表
coLumn_ names = data. columns. tolist( )
#列名列表删除"CLass"
col umn_ names. remove( '该次活动中是否有使用优惠券')
#获取特征数据
features = data[coLumn_ names]. values
#拆分数据集
# 20%作为测试集，其余作为训练集
X_ train, x_ test, y_ .train, y_ test = train_ test_ split(features, target, t
est_ size=0.2, stratify=target)
#找到树的最佳深度
#设定树的深度范围
depth_ range = np. arange(1, 20)
true_ score_ List = [
test_ score_ List = []
for d in depth_ range:
clf = Decis ionTreeClassifier(max depth=d) .fit(X_ train, y_ train)
true_ score = clf. score(X_ .train, y_ .train)
true_ score_ list. append( true_ score)
test_ score = clf.score(X_ test, y_ test)
test_ score_ list. append(test_ score)
plt. figure(figsize=(6, 4), dpi=120)
plt.grid()
plt. xLabeL( 'max tree deep ')
plt. ylabel( 'score ')
plt. plot(depth_ range, test_ score_ list, LabeL='test score')
plt. plot(depth_ range, true_ score_ list, Label= 'train score ')
plt. Legend()
plt. show()
#获取测试数据集评分最高的索引
te_ best_ index = int(np. argmax(test_ score_ list))
#树的高度=测试数据集评分最高的索引+1
tree_ dep = te_ best_ index + 1
print('树的最佳深度，:，tree_ dep) 
#建立模型
model = Dec is ionTreeClassifier(max_ depth=tree_ dep)
#训练模型
model.fit(X_ train, y_ train)
#模型预测训练集的数据
y_ pred = modeL. predict(X_ test)
#模型评估
#分类指标的文本报告
print('分类指标的文本报告: ')
print(cLassification_ report(y_ test, y_ pred))
#列出决策树的所有标签，是一个数组
class_ names = model. classes_
#将标签类型转为str
class_ names = [str(i) for i in clLass_ names]
#把这棵树modeL进行图像化，让人看着更加清晰
dot_ data = tree. export_ graphviz (model, feature names=coLumn_ names, # 类
别名称
class_ names=class_ names, # 特征名称
filled=True, # 给图形填充颜色
rounded=True, # 图形的节点是圆角矩形
special_ characters=True, #不忽略特殊字
符
max_ depth=tree_ dep) # 表示的最大深度。
graph = pydotplus . graph_ from_ dot_ data(dot_ data)
graph.write_ png('决策树.png') # 保存图像
importances = pd. DataFrame({ 'feature': column_ names, ' importance': np.r
ound(model . feature_ importances_ ，3)})
importances = importances . sort_ values( 'importance', ascending=False)
importances.to_ csv(' 属性重要性排
序.csv'，index=False, encoding='utf-8-sig')
#获取预测数据
file_ pre = ' 优惠券预测数据. csv'
df_ pre = pd.read csv(file_ pre)
#模型预测
y_ predict = model . predict(df_ pre. values)
df_ pre['labeL'] = y_ predict
df_ pre. to_ CsV( '预测结果数据.csv', index=False)
