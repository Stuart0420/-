#!/usr/bin/env python
# coding: utf-8

# In[49]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.sans-serif'] = ['SimHei']


# In[50]:


df = pd.read_csv('data/titanic.csv')
df.head(50)


# In[51]:


# 封装查看缺失值的热力图函数
def check_NullHeatMap(df):
    sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap = 'viridis')
    plt.show()


# In[52]:


# 查看数据样本特点
df.describe()


# In[53]:


# 检查缺失数据点
check_NullHeatMap(df)


# In[54]:



import scipy.stats as ss

original_age_normaltest = ss.normaltest(df['Age'].dropna())

plt.figure(figsize = (14, 6))

plt.subplot(1, 2, 1)
plt.hist(df['Age'].dropna(), bins = 30)
plt.title('年龄分布')

plt.show()


# In[55]:


def isNormal(data, significance_level = 0.05):
    _, p_value = ss.normaltest(data.dropna())
    return p_value > significance_level


data = df['Age']
result = isNormal(data)

if result:
    print("数据服从正态分布")
else:
    print("数据不服从正态分布")


# In[56]:


# 检查年龄缺失值数量
# 处理年龄缺失值
missing_age = df['Age'].isnull().sum()
missing_age


# In[57]:


# 填充年龄的缺失值为均值
df['Age'][df['Age'].isnull()] = df['Age'].mean()
df['Age'][df['Age'].isnull()] = df['Age'].mean()


# In[58]:


df


# In[59]:


df.describe()


# In[60]:


# 再次检查年龄缺失值数量
missing_age = df['Age'].isnull().sum()
missing_age


# In[61]:


check_NullHeatMap(df)


# In[62]:


# 删除Cabin列中的缺失值
df.drop('Cabin', axis = 1, inplace = True)
df.dropna(inplace = True)
df


# In[63]:


check_NullHeatMap(df)


# In[64]:


# 检查离群点
sns.swarmplot(data = df, x = 'Embarked', y = 'Fare', hue = 'Survived', )
plt.title("费用+登船口+生还结果关系图")
plt.show()


# In[65]:


# 看到C港口出现了3个船票最大值的点，检查下是否存在离群点
# 绘制箱型图
sns.boxenplot(data = df, x = 'Embarked', y = 'Fare', hue = 'Survived', )
plt.show()


# In[66]:


# 船票存在较大的峰态指数
df['Fare'].skew()


# In[67]:


# 绘制直方图
sns.distplot(df['Fare'], color = 'r')
plt.show()


# In[68]:


# 船票费用偏左
df['Fare'].describe()


# In[69]:


"确定船票为离群数，但是不能删除，因为只有两项数据同时是离群才能被删除"


# In[70]:


# 数据清洗，准备训练模型以及roc曲线
# 先把性别转化了，0/1来表示
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)
# 将Embarked转化成0,1代表S,,,00代表C,,10代表Q
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)


# In[71]:


X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']
# 数据拆分, 测试集分20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

# 训练模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# ROC曲线
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# 绘制混淆矩阵
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
sns.heatmap(conf_matrix, annot=True, fmt='g')
plt.title('混淆矩阵')
plt.xlabel('精确率')
plt.ylabel('实际情况')

# 绘制ROC曲线
plt.subplot(1, 2, 2)
plt.plot(fpr, tpr, label='AUC = {:.2f}'.format(roc_auc))
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC 曲线')
plt.xlabel('错误率')
plt.ylabel('正确率')
plt.legend(loc='lower right')

plt.tight_layout()
plt.show()


# In[73]:


# evaluate模型
# 计算度量标准
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)

# 打印度量标准
print("准确率:", accuracy)
print("精确率:", precision)
print("召回率:", recall)
print("F1分数", f1)
print("AUC分数", auc_score)
# 准备绘图数据
metrics = ['准确率', '精确率', '召回率', 'F1分数', 'AUC分数']
values = [accuracy, precision, recall, f1, auc_score]

# 绘制评估指标条形图
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
sns.barplot(x=metrics, y=values)
plt.title('模型评估')
plt.ylabel('效果')
plt.show()


# In[ ]:




