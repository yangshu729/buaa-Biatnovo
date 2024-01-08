import numpy as np
import pandas as pd

def sort_by_two_parts(train_groups):
    # 修改代码：先拆分 spec_group_id，然后排序
    train_set = pd.concat(train_groups)
    train_set[['group', 'id']] = train_set['spec_group_id'].str.extract(r'([A-Za-z]+\d*):(\d+)')
    train_set['id'] = train_set['id'].astype(int)  # 转换为整数以便排序
    train_set = train_set.sort_values(by=['group', 'id'])
    # 如果不需要临时列，可以在排序后删除它们
    train_set = train_set.drop(['group', 'id'], axis=1)
    return train_set


df = pd.read_csv("/home/azureuser/biatnovo/train-data/ftp.peptideatlas.org/biatNovo/training.feature.csv")
# 按seq字段分组
grouped = df.groupby('seq')

# 获取所有组的列表
groups = [group for _, group in grouped]

# 打乱组的顺序
np.random.shuffle(groups)

# 计算各数据集的大小
num_train = int(len(groups) * 0.9)
num_valid = int(len(groups) * 0.05)

# 分配组到训练集、验证集和测试集
train_groups = groups[:num_train]
valid_groups = groups[num_train:num_train + num_valid]
test_groups = groups[num_train + num_valid:]

# 合并组成最终的数据集
train_set = sort_by_two_parts(train_groups)
valid_set = sort_by_two_parts(valid_groups)
test_set = sort_by_two_parts(test_groups)

# 输出数据集大小
print("Training set size:", len(train_set))
print("Validation set size:", len(valid_set))
print("Test set size:", len(test_set))
train_set.to_csv("/home/azureuser/biatnovo/train-data/ftp.peptideatlas.org/biatNovo/train_dataset_unique.csv", index=False)
valid_set.to_csv("/home/azureuser/biatnovo/train-data/ftp.peptideatlas.org/biatNovo/valid_dataset_unique.csv", index=False)
test_set.to_csv("/home/azureuser/biatnovo/train-data/ftp.peptideatlas.org/biatNovo/test_dataset_unique.csv", index=False)