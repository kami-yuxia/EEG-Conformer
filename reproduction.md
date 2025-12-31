2025.12.27
1. dataset_download.py下载了.npy格式BCI_competition_IV2a数据集到dataset里
2. npy_to_mat.py将.npy格式数据集转换为.mat格式数据集,并且切除了一个无效点
3. 修改了conformer.py的self.root为'dataset_mat/'
4. conda activate eeg_conformer

2025.12.28-29
1. 运行conformer.py, 结果保存在results/baseline文件夹里
2. 对SO6,seed1674运行LabelSmoothingCrossEntropyConformer.py, 结果保存在results/LabelSmoothingCrossEntropyConformer文件夹里
3. 对SO6,seed1674运行MishConformer.py, 结果保存在results/MishConformer文件夹里
4. 对SO6,seed1674运行SEBlockConformer.py, 结果保存在results/SEBlockConformer文件夹里
5. 结果对比如下：
   - baseline: 0.6458333333333334
   - LabelSmoothingCrossEntropyConformer: 0.6631944444444444
   - MishConformer: 0.6666666666666666
   - SEBlockConformer: 0.6458333333333334

12.30
1. 对SO6,seed542进行对比实验验证mish,labelsmooth是不是真的还有促进效果,发现mish有促进效果,而labelsmooth没有,至少说明labelsmooth的效果不好
2. 为了进一步验证mish的促进效果,对所有对象都运行MishConformer.py,且seed与baseline相同,结果保存在results/mish文件夹里

12.31
1. 复现消融实验,对比NOTransformer.py,NOaugmentation.py与原baseline的结果
2. NOTransformer.py注释掉TransformerEncoder, 结果保存在results/NOTransformer文件夹里
3. NOaugmentation.py注释掉数据增强, 结果保存在results/NOaugmentation文件夹里
```python
# data augmentation(下面代码注释掉)
aug_data, aug_label = self.interaug(self.allData, self.allLabel)
img = torch.cat((img, aug_data))
label = torch.cat((label, aug_label))
```