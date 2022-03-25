import numpy as np #数据处理库
import pandas as pd # pandas :数据分析库
from sklearn.model_selection import train_test_split  # sklearn:机器学习库
import matplotlib.pyplot as plt # matplotlib：数据可视化库
from sklearn.linear_model import LinearRegression
import warnings   # 发出警告,或者忽略它或引发异常。
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['simhei']
from sklearn import metrics

#二 获取数据
file_name = '空调销售量数据.csv'
df = pd.read_csv(file_name)

#三、数据预处理
# 数据存在空值, 删除空值
df = df.dropna()
# 删除完全一样的数据，去重
df.drop_duplicates(inplace=True)  # inplace :是直接在原来数据上修改还是保留一个副本

#四、拆分数据集：
X = df[['电视广告费用(万元)', '互联网广告费用(万元)', '收音机广告费用(万元)','报纸广告费用(万元)']]
y = df['销售量(万台)']
#  拆分数据集，一部分作为训练集，一部分作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=100)

#五 建立模型
lr = LinearRegression()

#六 训练模型
lr.fit(X_train, y_train)

#七、评估模型

y_pred = lr.predict(X_test)
mse_value = metrics.mean_squared_error(y_test, y_pred)
print('均方误差MSE:', mse_value)
r2_score_value = metrics.r2_score(y_test, y_pred)
print('决定系数R2:', r2_score_value)

# 测试集：可视化测试集的真实数据与模型预测的测试集数据
fig, axs = plt.subplots(figsize=(9, 3))
plt.plot(range(1, len(y_test) + 1), y_test, 's-', color='orangered', label="真实值", linewidth=2)  # s-:方形
plt.plot(range(1, len(y_test) + 1), y_pred, 'o-', color='dodgerblue', label="预测值", linewidth=2)  # o-:圆形
plt.ylabel("价格", fontsize=16)  # 纵坐标名字
plt.legend(loc="best")  # 图例
plt.savefig('模拟值与测试值的关系.png')
plt.show()


#八、输出模型
print('权重参数：',lr.coef_)  # 得到[\beta _{1},\beta _{2},...\beta _{k}]
print('截距：',lr.intercept_ )# \beta _{0}，截距，默认有截距
f = str(lr.intercept_) + ' + ' + str(lr.coef_[0]) + '*电视广告费用(万元)' + ' + ' + str(lr.coef_[1]) +'*互联网广告费用(万元)' + ' + ' + str(lr.coef_[2]) + '*收音机广告费用(万元)' + ' + ' + str(lr.coef_[3]) + '*报纸广告费用(万元)'
print('模型：y=', f)


#九、模型预测
# 获取下期因素数据
df_next = pd.read_csv('空调销售量预测数据.csv')
# 模型预测下期价格数据
price_next = lr.predict(df_next[['电视广告费用(万元)', '互联网广告费用(万元)','收音机广告费用(万元)','报纸广告费用(万元)']].values)
# 保存预测结果
df_next['预测值'] = price_next
df_next.to_csv('预测数据结果.csv',  index=False, encoding='utf-8-sig')
