#一、导入Python库文件
# 导入Python库
import pandas as pd  # pandas :数据分析库
from sklearn.model_selection import train_test_split  # sklearn:机器学习库
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt  # matplotlib：数据可视化库
import matplotlib as mpl
from sklearn.metrics import roc_curve, auc
from sklearn import tree

import pydotplus

import graphviz  # 绘图工具

import numpy as np


plt.rcParams['font.sans-serif'] = ['simhei']  # 中文显示
mpl.rcParams['axes.unicode_minus'] = False  # 负号显示
import warnings  # 发出警告,或者忽略它或引发异常。
warnings.filterwarnings('ignore')


#二、获取数据
# 获取信用卡欺诈数据
data = pd.read_csv('基于决策树识别信用卡数据欺诈行为/信用卡欺诈数据.csv')


#三、数据预处理
# 数据存在空值, 删除空值
data = data.dropna()
# 删除完全一样的数据，去重
data = data.drop_duplicates()


#四、查看数据分布

# 数据分析：统计欺诈非欺诈数量
num_1 = len(data[data['Class']==1])
num_0 = len(data[data['Class']==0])
print('欺诈的数量：', num_1)
print('非欺诈的数量：', num_0)


五、拆分数据集
# 构造训练数据
# 获取类别数据
target = data['Class'].values
# 获取列名列表
column_names = data.columns.tolist()
# 列名列表删除“Class"
column_names.remove('Class')
# 获取特征数据
features = data[column_names].values
# 拆分数据集
# 20%作为测试集，其余作为训练集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, stratify=target)


#六、寻找树的最佳深度

# 找到树的最佳深度
# 设定树的深度范围
depth_range = np.arange(1, 20)
true_score_list = []
test_score_list = []
for d in depth_range:
    clf = DecisionTreeClassifier(max_depth=d).fit(X_train, y_train)
    true_score = clf.score(X_train, y_train)
    true_score_list.append(true_score)
    test_score = clf.score(X_test, y_test)
    test_score_list.append(test_score)

plt.figure(figsize=(6, 4), dpi=120)
plt.grid()
plt.xlabel('max tree deep')
plt.ylabel('score')
plt.plot(depth_range, test_score_list, label='test score')
plt.plot(depth_range, true_score_list, label='train score')
plt.legend()
plt.show()

# 获取测试数据集评分最高的索引
te_best_index = int(np.argmax(test_score_list))
# 树的高度=测试数据集评分最高的索引+1
tree_dep = te_best_index + 1
print('树的最佳深度，：', tree_dep)


#七、建立模型
# 建立模型
model = DecisionTreeClassifier(max_depth=tree_dep)


#八、训练模型
# 训练模型
model.fit(X_train, y_train)


#九、模型评估
y_pred = model.predict(X_test)
# 模型评估
# 分类指标的文本报告

print('分类指标的文本报告:')

print(classification_report(y_test, y_pred))

#十、可视化决策树
# 列出决策树的所有标签，是一个数组
class_names = model.classes_
# 将标签类型转为str
class_names = [str(i) for i in class_names]
# 把这棵树model进行图像化，让人看着更加清晰
dot_data = tree.export_graphviz(model, feature_names=column_names,  # 类别名称
                                class_names=class_names,  # 特征名称
                                filled=True,  # 给图形填充颜色
                                rounded=True,  # 图形的节点是圆角矩形
                                special_characters=True,  # 不忽略特殊字符
                                max_depth=tree_dep)  # 表示的最大深度。
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_png('基于决策树识别信用卡数据欺诈行为/决策树.png')  # 保存图像

importances = pd.DataFrame({'feature': column_names, 'importance': np.round(model.feature_importances_, 3)})
importances = importances.sort_values('importance', ascending=False)
importances.to_csv('基于决策树识别信用卡数据欺诈行为/属性重要性排序.csv', index=False, encoding='utf-8-sig')


#十一、模型预测
# 获取预测数据
file_pre = '基于决策树识别信用卡数据欺诈行为/信用卡欺诈预测数据.csv'
df_pre = pd.read_csv(file_pre)
# 模型预测
y_predict = model.predict(df_pre.values)
df_pre['label'] = y_predict
df_pre.to_csv('基于决策树识别信用卡数据欺诈行为/预测结果数据.csv', index=False)

