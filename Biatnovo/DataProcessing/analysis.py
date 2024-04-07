import pandas as pd

# 读取CSV文件
df1 = pd.read_csv('~/biatnovo/train-data/ftp.peptideatlas.org/biatNovo/train_dataset_unique.csv')
print(len(df1))
df2 = pd.read_csv('/root/biatnovo/DeepNovo-DIA/oc/oc_test.feature.csv')
print(len(df2))

# 提取两个数据框中唯一的seq值
unique_seqs_df1 = set(df1['seq'].unique())
unique_seqs_df2 = set(df2['seq'].unique())

# 找到公共的seq
common_seqs = unique_seqs_df1.intersection(unique_seqs_df2)

# 计算df2中属于公共seq的条目占df2整体的比例
# 首先，创建一个指示列，标记每个条目的seq是否属于公共seq
df2['is_common_seq'] = df2['seq'].apply(lambda x: x in common_seqs)

# 然后，统计属于公共seq的条目数
common_seq_count_in_df2 = df2['is_common_seq'].sum()

# 计算比例
common_seq_proportion_in_df2 = common_seq_count_in_df2 / len(df2)

# 输出结果
print(f"在第二个CSV文件中，属于公共seq的条目数量: {common_seq_count_in_df2}")
print(f"这些条目占第二个CSV文件总条目的比例: {common_seq_proportion_in_df2:.2%}")