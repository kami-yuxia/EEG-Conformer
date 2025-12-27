import os
import numpy as np
from moabb.datasets import BNCI2014001
from moabb.paradigms import MotorImagery
from sklearn.preprocessing import LabelEncoder

# 1. 创建保存数据的文件夹
data_dir = './dataset'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

print(f">>> 准备开始下载 BCI Competition IV 2a 所有 9 名受试者的数据...")
print(f">>> 数据将保存在: {os.path.abspath(data_dir)}")

# 2. 初始化数据集工具
# MOABB 会自动管理下载，如果本地已经有了（比如受试者1），它会跳过下载直接读取
dataset = BNCI2014001()
paradigm = MotorImagery(n_classes=4, fmin=4, fmax=40)

# 3. 循环处理受试者 1 到 9
for subject_id in range(1, 10):
    print(f"\n[{subject_id}/9] 正在处理受试者 {subject_id} ...")
    
    try:
        # 指定当前受试者
        dataset.subject_list = [subject_id]
        
        # 获取数据 (这一步会自动下载)
        X, y, metadata = paradigm.get_data(dataset=dataset, subjects=[subject_id])
        
        # 标签从单词转数字 (Label Encoding)
        le = LabelEncoder()
        y = le.fit_transform(y)
        
        # 简单检查一下
        print(f"    - 数据形状: {X.shape}")
        print(f"    - 标签示例: {y[:5]}")
        
        # 切分训练集和测试集 (Session T 和 Session E)
        # BCI IV 2a 刚好前一半是训练(T)，后一半是测试(E)
        mid = len(y) // 2
        X_train, y_train = X[:mid], y[:mid]
        X_test, y_test = X[mid:], y[mid:]
        
        # 构造文件名 (例如 A01T_data.npy, A02T_data.npy ...)
        sub_code = f'A0{subject_id}'
        
        # 保存文件
        np.save(os.path.join(data_dir, f'{sub_code}T_data.npy'), X_train)
        np.save(os.path.join(data_dir, f'{sub_code}T_label.npy'), y_train)
        np.save(os.path.join(data_dir, f'{sub_code}E_data.npy'), X_test)
        np.save(os.path.join(data_dir, f'{sub_code}E_label.npy'), y_test)
        
        print(f"    - 成功保存 {sub_code} 系列文件！")
        
    except Exception as e:
        print(f"    - ❌ 受试者 {subject_id} 处理失败: {e}")

print("\n>>> 🎉 所有数据下载及处理完毕！")