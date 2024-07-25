import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def get_csv_file_paths(folder_path):
    # 使用 glob 模块查找所有的 CSV 文件
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"Found {len(csv_files)} CSV files in the folder '{folder_path}'.")
    return csv_files

def remove_outliers_iqr(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    # print(Q1)
    # print("------------------------------")
    # print(Q3)
    IQR = Q3 - Q1
    multiplier = 100.0
    return data[~((data < (Q1 - multiplier * IQR)) | (data > (Q3 + IQR * multiplier))).any(axis=1)]

def smooth_data(data, window_size):
    smoothed_data = data.rolling(window=window_size, center=True).mean()
    padding = (window_size - 1) // 2
    smoothed_data = smoothed_data.fillna(method='bfill').fillna(method='ffill')
    return smoothed_data

def calculate_dkappa(t, kappa, segment_size=10):
    dkappa_diff = []
    t_diff = []

    for i in range(0, len(t) - segment_size, segment_size):
        t_segment = t[i:i + segment_size + 1]
        kappa_segment = kappa[i:i + segment_size + 1]

        dkappa_segment = np.diff(kappa_segment) / np.diff(t_segment)
        
        # 取第一个时间点和对应的速度值
        t_diff.append(t_segment[0])
        dkappa_diff.append(dkappa_segment[0])

    return np.array(t_diff), np.array(dkappa_diff)

def calculate_str_speed(t, str_angle):
    str_speed = np.diff(str_angle) / np.diff(t)
    t_diff = t[:-1]
    return t_diff, str_speed

def plot(file_paths, value_y, flag_smooth):
    combined_data = pd.DataFrame()
    
    for file_path in file_paths:
        data = pd.read_csv(file_path, low_memory=False)
        if value_y == 'kappa':
            values = data.iloc[:, 5]
        elif value_y == 'str_angle':
            values = data.iloc[:, 13]
        elif value_y == 'yaw_v':
            values = data.iloc[:, 18]
        elif value_y == 'dkappa':
            t = data.iloc[:, 7]
            kappa = data.iloc[:, 5]
            speeds = data.iloc[:, 3]

            data_frame = pd.DataFrame({'Time': t, 'Kappa': kappa, 'Speed': speeds})
            data_frame['Time'] = pd.to_numeric(data_frame['Time'], errors='coerce')
            data_frame['Kappa'] = pd.to_numeric(data_frame['Kappa'], errors='coerce')
            data_frame['Speed'] = pd.to_numeric(data_frame['Speed'], errors='coerce')
            data_frame.dropna(inplace=True)

            t_diff, dkappa_diff = calculate_dkappa(data_frame['Time'].values, data_frame['Kappa'].values)
            speed_diff = data_frame['Speed'].values[::10]

            # 确保速度数据与 dkappa 数据长度一致
            if len(speed_diff) > len(dkappa_diff):
                speed_diff = speed_diff[:len(dkappa_diff)]
            elif len(dkappa_diff) > len(speed_diff):
                dkappa_diff = dkappa_diff[:len(speed_diff)]

            data_frame = pd.DataFrame({'Speed': speed_diff, 'Value': dkappa_diff, 'Time': t_diff})
            combined_data = pd.concat([combined_data, data_frame])
            continue
        elif value_y == 'str_speed':
            t = data.iloc[:, 7]
            str_angle = data.iloc[:, 13]
            speeds = data.iloc[:, 3]

            data_frame = pd.DataFrame({'Time': t, 'Str_Angle': str_angle, 'Speed': speeds})
            data_frame['Time'] = pd.to_numeric(data_frame['Time'], errors='coerce')
            data_frame['Str_Angle'] = pd.to_numeric(data_frame['Str_Angle'], errors='coerce')
            data_frame['Speed'] = pd.to_numeric(data_frame['Speed'], errors='coerce')
            data_frame.dropna(inplace=True)

            t_diff, str_speed = calculate_str_speed(data_frame['Time'].values, data_frame['Str_Angle'].values)
            speed_diff = data_frame['Speed'].values[:-1]

            data_frame = pd.DataFrame({'Speed': speed_diff, 'Value': str_speed})
            combined_data = pd.concat([combined_data, data_frame])
            continue
        else:
            print("Invalid Value. Please Check!")
            return
        
        speeds = data.iloc[:, 3]
        data_frame = pd.DataFrame({'Speed': speeds, 'Value': values})
        data_frame['Speed'] = pd.to_numeric(data_frame['Speed'], errors='coerce')
        data_frame['Value'] = pd.to_numeric(data_frame['Value'], errors='coerce')
        data_frame.dropna(inplace=True)
        
        combined_data = pd.concat([combined_data, data_frame])
    
    # combined_data = remove_outliers_iqr(combined_data)
    remove_outliers_iqr(combined_data)
    sorted_data = combined_data.sort_values(by='Speed')
    print(combined_data.shape)
    step_size = 10
    sparse_sorted_data = sorted_data.iloc[::step_size, :]
    
    min_speed = sorted_data['Speed'].min()
    max_speed = sorted_data['Speed'].max()
    speed_intervals = np.arange(0, max_speed, 0.05)
    
    max_values = []
    min_values = []
    valid_speed_intervals = []
    
    for speed in speed_intervals:
        subset = sorted_data[(sorted_data['Speed'] >= speed) & (sorted_data['Speed'] < speed + 0.05)]
        if len(subset) > 0:
            max_value = subset['Value'].max()
            min_value = subset['Value'].min()
            if max_value == min_value:
                continue
            max_values.append(max_value)
            min_values.append(min_value)
            valid_speed_intervals.append(speed)
    
    plt.figure(figsize=(13, 7))
    plt.scatter(sparse_sorted_data['Speed'], sparse_sorted_data['Value'], s=0.8, alpha=0.3, label='Data Points')
    if(flag_smooth):
        max_value_smoothed = smooth_data(pd.Series(max_values), window_size=10).to_numpy()
        min_value_smoothed = smooth_data(pd.Series(min_values), window_size=10).to_numpy()
        plt.plot(valid_speed_intervals, max_value_smoothed, color='blue', alpha=0.5, marker='o', linestyle='-', linewidth=1, label='Max Value')
        plt.plot(valid_speed_intervals, min_value_smoothed, color='red', alpha=0.5, marker='o', linestyle='-', linewidth=1, label='Min Value')

    else:
        plt.plot(valid_speed_intervals, max_values, color='blue', alpha=0.5, marker='o', linestyle='-', linewidth=1, label='Max Value')
        plt.plot(valid_speed_intervals, min_values, color='red', alpha=0.5, marker='o', linestyle='-', linewidth=1, label='Min Value')

    y_min = combined_data['Value'].min() - 0.05 * combined_data['Value'].max()
    y_max = combined_data['Value'].max() + 0.05 * combined_data['Value'].max()
    plt.ylim(y_min, y_max)
    plt.title('Max and Min ' + value_y + ' vs. Speed Intervals in Data_merge')
    plt.xlabel('Speed (m/s)')
    plt.ylabel(value_y)
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    folder_path = '/home/xyf/Downloads/Data_merge'
    file_paths = get_csv_file_paths(folder_path)

    flag_smooth = True
    parameter = 'dkappa'
    if file_paths:
        plot(file_paths, parameter, False)
        plot(file_paths, parameter, True)
    else:
        print("No CSV files found in the specified folder.")