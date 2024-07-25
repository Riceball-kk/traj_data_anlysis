import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_dkappa(t, kappa):
    # 方法一：一阶差分
    dkappa_diff = np.diff(kappa) / np.diff(t)
    t_diff = t[:-1]  # 对应的时间戳
    return t_diff, dkappa_diff

def remove_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    multiplier=10.0
    return data[~((data < (Q1 - multiplier * IQR)) | (data > (Q3 + multiplier * IQR))).any(axis=1)]

def smooth_data(data, window_size):
    smoothed_data = data.rolling(window=window_size, center=True).mean()
    padding = (window_size - 1) // 2
    smoothed_data = smoothed_data.fillna(method='bfill').fillna(method='ffill')
    return smoothed_data

def plot_dkappa_vs_speed(file_path):
    data = pd.read_csv(file_path, low_memory=False)

    # 选择时间、曲率和速度列
    t = data.iloc[:, 7]  # 第8列
    kappa = data.iloc[:, 5]  # 第6列
    speeds = data.iloc[:, 3]  # 第4列

    # 创建数据框
    data_frame = pd.DataFrame({'Time': t, 'Kappa': kappa, 'Speed': speeds})

    # 将无法转换的字符串替换为 NaN
    data_frame['Time'] = pd.to_numeric(data_frame['Time'], errors='coerce')
    data_frame['Kappa'] = pd.to_numeric(data_frame['Kappa'], errors='coerce')
    data_frame['Speed'] = pd.to_numeric(data_frame['Speed'], errors='coerce')

    # 删除包含NaN值的行
    data_frame.dropna(inplace=True)

    # 计算 dkappa
    t_diff, dkappa_diff = calculate_dkappa(data_frame['Time'].values, data_frame['Kappa'].values)
    speed_diff = data_frame['Speed'].values[:-1]

    # 将数据组合成DataFrame以便于去除离群值
    dkappa_speed_data = pd.DataFrame({'Speed': speed_diff, 'dkappa': dkappa_diff})

    # 去除离群值
    dkappa_speed_data_cleaned = remove_outliers_iqr(dkappa_speed_data)

    # 按速度大小排序
    sorted_data = dkappa_speed_data_cleaned.sort_values(by='Speed')

    # 稀疏化处理
    step_size = 10  # 每10个数据取一个
    sparse_sorted_data = sorted_data.iloc[::step_size, :]

    # 计算速度的最大值和最小值
    min_speed = sorted_data['Speed'].min()
    max_speed = sorted_data['Speed'].max()

    # 创建速度区间
    speed_intervals = np.arange(min_speed, max_speed, 0.05)
    print(speed_intervals.shape)

    # 计算每个速度区间的最大和最小 dkappa
    max_dkappas = []
    min_dkappas = []
    valid_speed_intervals = []

    for speed in speed_intervals:
        subset = sorted_data[(sorted_data['Speed'] >= speed) & (sorted_data['Speed'] < speed + 0.05)]
        if len(subset) > 0:
            max_dkappa = subset['dkappa'].max()
            min_dkappa = subset['dkappa'].min()
            if max_dkappa == min_dkappa:
                continue
            max_dkappas.append(max_dkappa)
            min_dkappas.append(min_dkappa)
            valid_speed_intervals.append(speed)  # 将有数据的速度区间添加到新的列表中

    # 对最大最小值进行平滑处理
    max_dkappas_smoothed = smooth_data(pd.Series(max_dkappas), window_size=8).to_numpy()
    min_dkappas_smoothed = smooth_data(pd.Series(min_dkappas), window_size=8).to_numpy()

    # 确保valid_speed_intervals的长度与平滑后的数据长度一致
    valid_speed_intervals = np.array(valid_speed_intervals[:len(max_dkappas_smoothed)])

    # 绘图
    plt.figure(figsize=(10, 6))

    # 绘制稀疏化处理后的 dkappa
    plt.scatter(sparse_sorted_data['Speed'], sparse_sorted_data['dkappa'], s=0.8, alpha=0.3, label='dkappa')

    # 绘制每个速度区间的最大和最小 dkappa
    plt.plot(valid_speed_intervals, max_dkappas_smoothed, color='blue', alpha=0.5, marker='o', linestyle='-', linewidth=1, label='Max dkappa')
    plt.plot(valid_speed_intervals, min_dkappas_smoothed, color='red', alpha=0.5, marker='o', linestyle='-', linewidth=1, label='Min dkappa')

    plt.title('Max and Min dkappa vs. Speed Intervals')
    plt.xlabel('Speed (m/s)')
    plt.ylabel('dkappa')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file_path = '/home/xyf/Downloads/Data_analysis/BYT-YMXQ.csv'  # 请替换为实际的文件路径
    plot_dkappa_vs_speed(file_path)
