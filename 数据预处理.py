# 缺失值处理
import pywt
from scipy.signal import butter, sosfilt
import warnings
import os
from torch import negative
import numpy as np
import matplotlib.pyplot as plt
flag = 0
for data_i, fp in data:
    missing_value_count = data_i.isnull().sum()
    if missing_value_count.sum() > 0:
        print(f'{fp} missing value count: {missing_value_count}')
        flag = 1
    else:
        print(f'{fp} missing value count: 0')
if flag == 0:
    print('All files have no missing value')

# 异常值处理
# 首先我们需要检查数据的异常值，这里我们使用箱线图来检查异常值
# 查看温度信号的分布情况
for data_i, fp in data:
    if data_i['温度信号'].min() < 0:
        # 第一次出现负值时的索引
        temperature = data_i['温度信号']
        temperature = np.array(list(set(temperature)))
        for index, negative_index in enumerate(np.where(temperature < 0)[0]):
            print(temperature[negative_index-10:negative_index+10])
            if index > 5:
                break
        # 可视化温度信号的序列分布
        plt.figure(figsize=(20, 5))
        plt.plot(data_i['温度信号'])
        plt.title(f'{fp} 温度信号')
        # 可视化温度信号的箱线图
        plt.figure(figsize=(20, 5))
        plt.boxplot(data_i['温度信号'])
        plt.title(f'{fp} 温度信号')
        plt.show()
        break

warnings.filterwarnings("ignore")

# 负值处理


def nega_outlier_process(data, col):
    # 数据最大索引
    index_max = data.shape[0]-1
    # 负值索引
    negaindex = pd.DataFrame()
    negaindex['index'] = data[data[col] < 0].index.tolist()

    # 查找多段负值分割点，用lix保存（‘负值索引’突变的索引）
    negaindex['ptp'] = negaindex['index'].rolling(
        2, min_periods=1).apply(np.ptp)
    lixmax = len(negaindex)-1
    lix = [0]+negaindex[negaindex['ptp'] > 1].index.tolist()+[lixmax]
    # print(negaindex,lix)
    # 循环读取每一个负值分段
    for i in range(len(lix)-1):
        # 最后一个分段处理，negai代表分段负值索引dataframe
        if lix[i+1] == lixmax:
            negai = negaindex['index'].iloc[lix[i]:]
        # 前面分段处理
        else:
            negai = negaindex['index'].iloc[lix[i]:lix[i+1]]
        # 分段内负值的极差
        nega_ptp = round(data[col][negai.tolist()].max() -
                         data[col][negai.tolist()].min(), 2)
        if negai.max() < index_max:
            # 分段两端正值的差（前提是有后分段）
            posi_ptp = round(data[col][negai.min()-1] -
                             data[col][negai.max()+1], 2)
            # 负值内部加absnega_max,同时加上正值前端
            data[col][negai.tolist()] = data[col][negai.tolist()] +\
                abs(data[col][negai.tolist()]).max() +\
                data[col][negai.min()-1]
            # 正值后端阶段加上nega_ptp,（st：posi_ptp大于0）再加posi_ptp
            if posi_ptp > 0:
                if lix[i+1] == lixmax:
                    data[col][negai.max()+1:] = data[col][negai.max()+1:] + \
                        nega_ptp+posi_ptp
                else:
                    data[col][negai.max()+1:negaindex['index'][lix[i+1]]] = data[col][negai.max() +
                                                                                      1:negaindex['index'][lix[i+1]]]+nega_ptp+posi_ptp
            else:
                if lix[i+1] == lixmax:
                    data[col][negai.max()+1:] = data[col][negai.max()+1:]+nega_ptp
                else:
                    data[col][negai.max()+1:negaindex['index'][lix[i+1]]
                              ] = data[col][negai.max()+1:negaindex['index'][lix[i+1]]]+nega_ptp
        else:
            # 如果没有后分段，负值内部加上nega_ptp,同时加上正值前端
            data[col][negai.tolist()] = data[col][negai.tolist()] +\
                abs(data[col][negai.tolist()]).max() +\
                data[col][negai.min()-1]
    return data


def outer_data(data, e):
    # 负值处理
    for col in ['累积量参数1', '累积量参数2', '部件工作时长']:
        if data[col].min() < 0:
            data = nega_outlier_process(data, col)
    data['温度信号'] = abs(data['温度信号'])
    if os.path.basename(e) == 'de5d7628aced6b08c5bc.csv':
        data['部件工作时长'][data['部件工作时长'] > 3000] = 1611.5
    if os.path.basename(e) == '21bade855e1f81d7e1c8.csv':
        data['部件工作时长'][data['部件工作时长'] == -4103.75] = -3.75
    return data


processed_data = []
for (df, e) in data:
    df = outer_data(df, e)
    processed_data.append((df, e))
assert len(processed_data) == len(data)
data = processed_data
data_length = len(data)


n_samples = data_length


def data_smooth(x, alpha=20, beta=1):
    new_x = np.zeros(x.shape[0])
    new_x[0] = x[0]
    for i in range(1, len(x)):
        tmp = x[i-1] * (alpha - beta) / alpha + x[i] * beta / alpha
        new_x[i] = x[i] - tmp
    return new_x


sample_duration = 0.02
sample_rate = n_samples * (1 / sample_duration)


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

# 高通


def high_pass_filter(x, low_cutoff=10000, sample_rate=sample_rate):
    nyquist = 0.5 * sample_rate
    norm_low_cutoff = low_cutoff / nyquist
    sos = butter(10, Wn=[norm_low_cutoff], btype='highpass', output='sos')
    filtered_sig = sosfilt(sos, x)
    return filtered_sig

# 降噪


def denoise_signal(x, wavelet='haar', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard')
                 for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')


# 高通滤波
# 可视化转速的时序分布图
plt.figure(figsize=(20, 10))
for (df, e) in data:
    for i in [
        '转速信号1',
        '转速信号2',
        '压力信号1',
        '压力信号2',
        '温度信号',
        '流量信号',
        '电流信号',
    ]:
        plt.figure(figsize=(20, 10))
        plt.plot(df[i], label=i)
        x_hp = high_pass_filter(
            df[i], low_cutoff=10000, sample_rate=sample_rate)  # 高通滤波
        x_dn = denoise_signal(x_hp, wavelet='haar', level=1)  # 降噪
        df_dn = df[i] - df[i].median()
        plt.figure(figsize=(20, 10))
        plt.plot(data_smooth(df_dn), label=i)
        plt.xlabel('time')
        plt.ylabel(i)
    plt.show()
    break
