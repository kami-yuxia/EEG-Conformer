2025.12.27
1. dataset_download.py下载了.npy格式BCI_competition_IV2a数据集到dataset里
2. npy_to_mat.py将.npy格式数据集转换为.mat格式数据集,并且切除了一个无效点
3. 修改了conformer.py的self.root为'dataset_mat/'
4. conda activate eeg_conformer