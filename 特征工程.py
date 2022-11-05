# 特征工程

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import string
import random
import matplotlib
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import sklearn

# 统计特征


def ptp(x):  # 极差
    return x.max()-x.min()


def margin(x):
    data = np.sqrt(abs(x))
    df_margin = (ptp(x)) / (pow(data.mean(), 2)+1e-10)
    return df_margin


def rms(x):
    data = np.sqrt(np.power(x.mean(), 2)+np.power(x.std(), 2))
    return data


def boxing(x):
    data = rms(x) / (np.abs(x.mean())+1e-10)
    return data


def peak(x):
    data = ptp(x) / (rms(x)+1e-10)


def pulse(x):
    data = ptp(x) / (np.abs(x.mean())+1e-10)
    return data


def soreoccurring(x):
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    counts[counts > 1] = 1
    return np.sum(counts * unique)


def pereoccurring_all(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    if x.size == 0:
        return np.nan
    value_counts = x.value_counts()
    reoccuring_values = value_counts[value_counts > 1].sum()
    if np.isnan(reoccuring_values):
        return 0
    return reoccuring_values / x.size


def datapoint(x):
    if len(x) == 0:
        return np.nan
    unique, counts = np.unique(x, return_counts=True)
    if counts.shape[0] == 0:
        return 0
    return np.sum(counts > 1) / float(counts.shape[0])

# 频域特征：信号傅里叶变换之后的均值，方差


def fft_mean(x):
    return np.mean(np.abs(np.fft.fft(x)))


def fft_std(x):
    return np.std(np.abs(np.fft.fft(x)))


# 用于训练的特征
def stat(data, c, name):
    c[name + '_max'] = data.max()  # 最大值
    c[name + '_min'] = data.min()  # 最小值
    c[name + '_ptp'] = ptp(data)  # 极差
    c[name + '_count'] = data.count()  # 非空值个数
    c[name + '_mean'] = data.mean()  # 均值
    c[name + '_std'] = data.std()  # 标准差
    c[name + '_skew'] = data.skew()  # 偏度 数据的不对称程度
    c[name + '_kurt'] = data.kurt()  # 峰度 数据分布顶的尖锐程度
    c[name + '_mode'] = data.mode()[0]  # 众数
    c[name + '_median'] = data.median()  # 中位数

    c[name + '_margin'] = margin(data)  # 裕度 表示数据的离散程度
    c[name + '_rms'] = rms(data)  # 均方根 均方根是一种对信号能量的度量，它是信号能量的平方根
    c[name + '_boxing'] = boxing(data)  # 波形
    c[name + '_pulse'] = pulse(data)  # 脉冲 峰值与均值的比值

    c[name + '_soreoccurring'] = soreoccurring(data)  # 重复出现的最大值
    c[name + '_pereoccurring_all'] = pereoccurring_all(data)  # 重复出现的比例
    c[name + '_datapoint'] = datapoint(data)  # 重复出现的数据点比例

    c[name + '_fft_mean'] = fft_mean(data)  # 傅里叶变换之后的均值
    c[name + '_fft_std'] = fft_std(data)  # 傅里叶变换之后的方差
    return c


# 输出所有设备类型
device = []
for data_i, f in train_set:
    if data_i['设备类型'].values[0] not in device:
        device.append(data_i['设备类型'].values[0])
print(device)
le = LabelEncoder()
le.fit(device)
device_le = le.transform(device)
# 建立设备类型与数字的映射
device_dict = dict(zip(device, device_le))
print(device_dict)


# PCA主成分分析


def process_sample_single_ratio(data, idx, ratio):
    lifemax = data['部件工作时长'].max()  # 最大工作时长
    temp_data = data[0:int(len(data)*ratio)]  # 取前ratio比例的数据作样本

    c = {'train_file_name': idx,
         '开关1_sum': temp_data['开关1信号'].sum(),
         '开关2_sum': temp_data['开关2信号'].sum(),
         '开关_sum': temp_data['开关1信号'].sum()+temp_data['开关2信号'].sum(),
         '告警1_sum': temp_data['告警信号1'].sum(),
         '告警1比例': data['告警信号1'].sum()/(1e-8+data['开关1信号'].sum()+data['开关2信号'].sum()),
         '设备': device_dict[data['设备类型'].values[0]],
         'life': lifemax - temp_data['部件工作时长'].max(),
         }
    for i in name_list[0:10]:
        c = stat(temp_data[i], c, i)
    this_tv_features = pd.DataFrame(c, index=[idx])

    return this_tv_features


# 处理训练集
train = process_sample_single_ratio(train_set[0][0], 0, ratio_list[0])
for i in range(len(ratio_list)):
    ratio = ratio_list[i]  # 取前ratio比例的数据作样本
    for j in range(len(train_set) // len(ratio_list)):
        if not(i == 0 and j == 0):
            temp = process_sample_single_ratio(train_set[i*160+j][0], j, ratio)
            train = pd.concat((train, temp))

assert len(train) == len(train_set), 'train length error'

# 因为describe输出的内容过大，我们将describe输出的内容保存在csv文件中进行观察
with open('train_describe.csv', 'w') as f:
    f.write(train.describe().to_csv())


feature_counts = train.shape[1]
print('feature_counts:', feature_counts)
# 分析不同特征与life的相关性，挑选出相关性较大的特征
corr = train.corr()
corr.to_csv('corr.csv')
corr['life'].sort_values(ascending=False)

# 删除相关性为nan的特征
del train['部件工作时长_min']
del train['累积量参数1_min']
del train['累积量参数2_min']


matplotlib.rc("font", family="MicroSoft YaHei", weight="bold")
# 绘制相关性热力图
corr_draw = train.corr()
cmap = sns.choose_diverging_palette()
f, ax = plt.subplots(figsize=(20, 20))  # 设置画布大小
mask = np.zeros_like(corr_draw, dtype=np.bool)  # 定义一个大小一致全为零的矩阵  用布尔类型覆盖原来的类型
mask[np.triu_indices_from(mask)] = True  # 返回矩阵的上三角，并将其设置为true
sns.heatmap(corr_draw, mask=mask  # 只显示为true的值
            , cmap=cmap  # , vmax=.3
            , center=0  # ,square=True
            # 底图带数字 True为显示数字
            , linewidths=.5, cbar_kws={"shrink": .5}, annot=False
            )
# heatmap = f.get_figure()
plt.savefig("D:/Data_Ana/HM/figs/corr_heatmap1.png",
            transparent=True, dpi=1200)


# 挑选相关性绝对值最小的十个特征
corr = train.corr()
corr_abs = corr['life'].abs().sort_values(ascending=False)[100:]
print(corr_abs)
# 删除corr_abs中的特征
for i in corr_abs.index:
    del train[i]
print(train.shape)


matplotlib.rc("font", family="MicroSoft YaHei", weight="bold")
# 绘制挑选之后的相关性热力图
corr_draw = train.corr()
cmap = sns.choose_diverging_palette()
f, ax = plt.subplots(figsize=(20, 20))  # 设置画布大小
mask = np.zeros_like(corr_draw, dtype=np.bool)  # 定义一个大小一致全为零的矩阵  用布尔类型覆盖原来的类型
mask[np.triu_indices_from(mask)] = True  # 返回矩阵的上三角，并将其设置为true
sns.heatmap(corr_draw, mask=mask  # 只显示为true的值
            , cmap=cmap  # , vmax=.3
            , center=0  # ,square=True
            # 底图带数字 True为显示数字
            , linewidths=.5, cbar_kws={"shrink": .5}, annot=False
            )
# heatmap = f.get_figure()
plt.savefig("D:/Data_Ana/HM/figs/corr_heatmap2.png",
            transparent=True, dpi=1200)

Y = train['life']
X = train.drop(['life'], axis=1)
print(X.shape, Y.shape)

ratio = random.randint(2, 98)/100
test = process_sample_single_ratio(test_set[0][0], 0, ratio)

for j in range(1, len(test_set)):
    ratio = random.randint(2, 98)/100
    temp = process_sample_single_ratio(test_set[j][0], j, ratio)  # print(temp)
    test = pd.concat((test, temp))

del test['部件工作时长_min']
del test['累积量参数1_min']
del test['累积量参数2_min']

# 删除corr_abs中的特征
for i in corr_abs.index:
    del test[i]

# 保存一下训练和测试数据，方便后续操作
train.to_csv("D:/Data_Ana/HM/data/train.csv")
test.to_csv("D:/Data_Ana/HM/data/test.csv")
train['life'].to_csv("D:/Data_Ana/HM/data/train_label.csv")

train = pd.read_csv("D:/Data_Ana/HM/data/train.csv")
test = pd.read_csv("D:/Data_Ana/HM/data/test.csv")
del train['Unnamed: 0']
del test['Unnamed: 0']

# 分段映射life,先绘制直方图

# 第一步
fcc_survey_df = train

# 对年龄特征进行分段标记：比如0-9分为0, 10-19为1....
# 先对年龄字典画直方图，直方图本身也是一种分段过程
# 第二步
f, ax = plt.subplots(figsize=(10, 10))
fcc_survey_df['life'].hist(color='#A9C5D3')
ax.set_xlabel('Life')
ax.set_ylabel('Frequency')
ax.set_title('Life bins')
# plt.show()
# plt.savefig("D:/Data_Ana/HM/figs/life_before.png", transparent=True,dpi = 600)

# 发现有异常值，处理后重新绘图
index = train[train.life == train['life'].max()].index.tolist()[0]
train = train.drop(index)
index = train[train.life == 0.].index.tolist()[0]
train = train.drop(index)
# 第一步
fcc_survey_df = train

# 对年龄特征进行分段标记：比如0-9分为0, 10-19为1....
# 先对年龄字典画直方图，直方图本身也是一种分段过程
# 第二步
f, ax = plt.subplots(figsize=(10, 10))
fcc_survey_df['life'].hist(color='#A9C5D3')
ax.set_xlabel('Life')
ax.set_ylabel('Frequency')
ax.set_title('Life bins')
# plt.show()
# plt.savefig("D:/Data_Ana/HM/figs/life_after.png",dpi = 600)

# 对剩下的特征进行pca降维
label = train['life']
TF_name_pca = train['train_file_name']
train_pca = train.copy()
del train_pca['life']
del train_pca['train_file_name']
feature_counts = train_pca.shape[1]
pca = PCA(n_components=int(feature_counts * 0.8))  # 保留80%的信息
pca.fit(train_pca)
train_pca = pca.transform(train_pca)
train_pca = pd.DataFrame(train_pca)
train_pca['life'] = label
train_pca['train_file_name'] = TF_name_pca
train = train_pca

# 用该映射矩阵处理test
label = test['life']
TF_name_pca = test['train_file_name']
test_pca = test.copy()
del test_pca['life']
del test_pca['train_file_name']
feature_counts = test_pca.shape[1]

test_pca = pca.transform(test_pca)
test_pca = pd.DataFrame(test_pca)
test_pca['life'] = label
test_pca['train_file_name'] = TF_name_pca
test = test_pca

matplotlib.rc("font", family="MicroSoft YaHei", weight="bold")
# 绘制PCA之后的相关性热力图
corr_draw = train.corr()
cmap = sns.choose_diverging_palette()
f, ax = plt.subplots(figsize=(20, 20))  # 设置画布大小
mask = np.zeros_like(corr_draw, dtype=np.bool)  # 定义一个大小一致全为零的矩阵  用布尔类型覆盖原来的类型
mask[np.triu_indices_from(mask)] = True  # 返回矩阵的上三角，并将其设置为true
sns.heatmap(corr_draw, mask=mask  # 只显示为true的值
            , cmap=cmap  # , vmax=.3
            , center=0  # ,square=True
            # 底图带数字 True为显示数字
            , linewidths=.5, cbar_kws={"shrink": .5}, annot=False
            )
# heatmap = f.get_figure()
plt.savefig("D:/Data_Ana/HM/figs/corr_heatmap_PCA.png",
            transparent=True, dpi=1200)
