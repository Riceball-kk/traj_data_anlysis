import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def calculate_left_side(L, kappa):
    return L * kappa

def calculate_right_side(str_angle, Rs):
    return np.tan(str_angle / 180 * math.pi / Rs)

def plot_verification(file_paths, L, Rs):
    combined_data = pd.DataFrame()

    for file_path in file_paths:
        data = pd.read_csv(file_path, low_memory=False)
        
        # 选择曲率和方向盘转角列
        kappa = data.iloc[:, 5]
        str_angle = data.iloc[:, 13]

        # 创建数据框
        data_frame = pd.DataFrame({'Kappa': kappa, 'Str_Angle': str_angle})

        # 将无法转换的字符串替换为 NaN
        data_frame['Kappa'] = pd.to_numeric(data_frame['Kappa'], errors='coerce')
        data_frame['Str_Angle'] = pd.to_numeric(data_frame['Str_Angle'], errors='coerce')

        # 删除包含NaN值的行
        data_frame.dropna(inplace=True)
        
        combined_data = pd.concat([combined_data, data_frame])
    
    # 计算等式两边
    left_side = calculate_left_side(L, combined_data['Kappa'])
    print(left_side.shape)
    right_side = calculate_right_side(combined_data['Str_Angle'], Rs)

    # 绘图
    plt.figure(figsize=(10, 10))
    plt.scatter(left_side, right_side, s=1, alpha=0.5)
    # plt.scatter(right_side, left_side, s=1, alpha=0.5)
    plt.plot([left_side.min(), left_side.max()], [left_side.min(), left_side.max()], color='red', label='y=x')
    plt.xlabel('L * Kappa')
    plt.ylabel('tan(Str_Angle / Rs)')
    plt.title('Verification of L * Kappa = tan(Str_Angle / Rs)')
    plt.grid(True)
    plt.show()
    

if __name__ == "__main__":
    file_paths = [
        '/home/xyf/Downloads/Data_FreeDriving/BYT-YMXQ_1.csv',
        '/home/xyf/Downloads/Data_FreeDriving/hxsgy-kxc.csv',
        '/home/xyf/Downloads/Data_FreeDriving/kxc-hxsgy.csv',
        '/home/xyf/Downloads/Data_FreeDriving/YMXQ_BYT.csv'
    ]
    L = 3.79  # 轴距
    Rs = 22  # 转动比
    plot_verification(file_paths, L, Rs)
