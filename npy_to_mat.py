import os
import numpy as np
import scipy.io

source_dir = './dataset'
target_dir = './dataset_mat'

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

print(f">>> [修复版] 正在将 .npy 转换为 .mat (强制截断为 1000 点)...")

for i in range(1, 10):
    sub_code = f'A0{i}'
    try:
        # 1. 读取数据
        train_data = np.load(os.path.join(source_dir, f'{sub_code}T_data.npy'))
        train_label = np.load(os.path.join(source_dir, f'{sub_code}T_label.npy'))
        test_data = np.load(os.path.join(source_dir, f'{sub_code}E_data.npy'))
        test_label = np.load(os.path.join(source_dir, f'{sub_code}E_label.npy'))

        # 2. 🔥【关键修复】强制截取前 1000 个时间点
        # 原始可能是 1001 或其他，原代码只接受 1000 (8*125)
        train_data = train_data[:, :, :1000]
        test_data = test_data[:, :, :1000]

        # 3. 维度转换: (Trial, Channel, Time) -> (Time, Channel, Trial)
        train_data_mat = train_data.transpose(2, 1, 0)
        test_data_mat = test_data.transpose(2, 1, 0)

        # 4. 标签处理: +1 以抵消原代码的 -1
        train_label_mat = train_label[:, np.newaxis] + 1
        test_label_mat = test_label[:, np.newaxis] + 1

        # 5. 保存
        scipy.io.savemat(os.path.join(target_dir, f'{sub_code}T.mat'), 
                         {'data': train_data_mat, 'label': train_label_mat})
        scipy.io.savemat(os.path.join(target_dir, f'{sub_code}E.mat'), 
                         {'data': test_data_mat, 'label': test_label_mat})
        
        print(f"    - [OK] {sub_code} (Shape: {train_data_mat.shape})")

    except FileNotFoundError:
        print(f"    - [跳过] 找不到 {sub_code}")

print("\n>>> ✅ 数据修复完成！现在数据长度严格为 1000，原代码可直接运行。")