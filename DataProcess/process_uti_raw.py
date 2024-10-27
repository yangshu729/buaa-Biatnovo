import pickle
import pandas as pd
import re
from collections import defaultdict

files = ["DIA_A5.mgf", "DIA_C10.mgf", "DIA_C2.mgf", "DIA_D1.mgf", "DIA_D2.mgf", "DIA_E11.mgf", "DIA_E5.mgf", "DIA_E6.mgf", "DIA_F10.mgf", "DIA_F6.mgf", "DIA_G10.mgf"]
# files = ["DIA_A5.mgf", "DIA_C10.mgf"]
# map_to_sample 根据文件名返回 FX
def map_to_sample(filename):
    try:
        # 找到文件在 files 列表中的索引（从 0 开始）
        index = files.index(filename)
        # 生成 F(index + 1)，因为下标从 1 开始
        return f'F{index + 1}'
    except ValueError:
        # 如果文件不在列表中，返回默认值
        return 'Unknown'
def merge_report_and_lib(report_tsv_path, report_lib_file):
    # 读取 report.tsv 文件
    report_df = pd.read_csv(report_tsv_path, sep='\t')

    # 读取 report-lib.tsv 文件
    report_lib_df = pd.read_csv(report_lib_file, sep="\t")

    # 创建一个从 ModifiedPeptide 到 PrecursorMz 的字典
    peptide_to_mz = report_lib_df.set_index("transition_group_id")["PrecursorMz"].to_dict()

    # 创建一个新的字段（列）来存储 PrecursorMz
    report_df["New_PrecursorMz"] = report_df["Precursor.Id"].map(peptide_to_mz)

    return report_df

import re
from collections import defaultdict

# 枚举24个窗口范围，前20个窗口大小为20，接下来2个为40，最后2个为60
window_ranges = [(400 + i * 20, 400 + (i + 1) * 20) for i in range(20)] + \
                [(800, 840), (840, 880)] + \
                [(880, 940), (940, 1000)]
print(window_ranges)

def parse_mgf(filepath):
    scan_map = {}  # 用于存储F组的scan信息
    current_scan = None
    current_rt = None
    current_f = None
    current_pepmass = None  # 存储PEPMASS值
    previous_pepmass = None  # 上一次的PEPMASS
    f_scan_start_index = {}  # 记录每个F组的开始下标

    
    with open(filepath, 'r') as f:
        for line in f:
            # 检查 PEPMASS=，提取PEPMASS的值，先记录下来
            if line.startswith("PEPMASS="):
                pepmass_values = line.strip().split("=")[1].split()  # 分割PEPMASS行
                current_pepmass = float(pepmass_values[0])  # 取PEPMASS的质量部分 (如650.0)

            # 检查 SCANS=，提取F组和XXX，处理PEPMASS与SCANS的关系
            elif line.startswith("SCANS="):
                scan_info = line.strip().split("=")[1]
                f_match = re.match(r'(F\d+):(\d+)', scan_info)  # 匹配F1:2, F2:3 等格式
                if f_match:
                    current_f = f_match.group(1)  # F组名 (如F1, F2)
                    scan_idx = int(f_match.group(2))  # 当前扫描的下标 (如2, 3)

                    # 如果该F组还没有记录开始下标，初始化它
                    if current_f not in f_scan_start_index:
                        f_scan_start_index[current_f] = scan_idx

                    # 检查PEPMASS是否出现了“跳跃”，从接近最大值跳到较小的值
                    if previous_pepmass and current_pepmass < 500 and previous_pepmass > 900:
                        # print(f"Notice: Full scan detected, resetting scan logic. Previous PEPMASS: {previous_pepmass}, Current PEPMASS: {current_pepmass}")
                        # 处理全scan逻辑，重置扫描下标
                        f_scan_start_index[current_f] = scan_idx

                    # 根据新的 scan_idx 和当前的窗口大小，确定low_mass和high_mass
                    relative_scan_idx = (scan_idx - f_scan_start_index[current_f]) % 24
                    low_mass, high_mass = window_ranges[relative_scan_idx]

                    # 更新 previous_pepmass
                    previous_pepmass = current_pepmass
                    current_scan = scan_info

            # 检查 RTINSECONDS=，提取retention time
            elif line.startswith("RTINSECONDS="):
                current_rt = float(line.strip().split("=")[1])

            # 检查 END IONS，代表扫描结束，此时需要记录扫描信息
            elif line.startswith("END IONS"):
                # 检查PEPMASS是否在low_mass和high_mass之间
                if current_pepmass is not None and low_mass <= current_pepmass <= high_mass:
                    # 如果当前扫描信息存在且PEPMASS在窗口内，将其添加到map中
                    if current_f is not None and current_scan is not None:
                        # 记录每个F组的最后一个扫描信息
                        scan_map[current_scan] = (current_rt, (low_mass, high_mass))
                        # print(f"Added scan {current_scan} to {current_f} with RT {current_rt} and PEPMASS {current_pepmass} in range {low_mass}-{high_mass}")
                    else:
                        # 如果当前扫描信息不存在，记录错误
                        print(f"Error: Current scan {current_scan} or F group {current_f} is None for PEPMASS {current_pepmass}")
                else:
                    # 如果PEPMASS不在范围内，记录错误或忽略该扫描
                    print(f"Warning: PEPMASS {current_pepmass} is out of range {low_mass}-{high_mass} for {current_scan}")

                # 重置当前的扫描信息，但不影响下次记录
                current_scan = None
                current_rt = None
                current_pepmass = None

        return scan_map

# 将scan_map保存到文件（序列化）
def save_scan_map(scan_map, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(scan_map, f)
    print(f"scan_map has been saved to {file_path}")

# 从文件中读取scan_map（反序列化）
def load_scan_map(file_path):
    with open(file_path, 'rb') as f:
        scan_map = pickle.load(f)
    print(f"scan_map has been loaded from {file_path}")
    return scan_map




def relate_scans_with_report(report_df, scan_map):
    
    # 初始化一个空的列表用于存储与每行关联的scan字符串
    related_scans_list = []
    # Create a dictionary to group scans by their 'F' prefix
    grouped_scan_map = defaultdict(list)
    for scan, values in scan_map.items():
        prefix = scan.split(":")[0]
        grouped_scan_map[prefix].append((scan, values))
    
    # Sort the scans within each group
    for prefix, scans in grouped_scan_map.items():
        grouped_scan_map[prefix] = sorted(scans, key=lambda x: x[1][0])

    total_rows = len(report_df)

     # First loop: iterate through each row in report.csv
    for index, row in report_df.iterrows():
        related_scans = []

        # Print progress after every 100 rows
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{total_rows} rows.")
    
        filename = map_to_sample(row["Run"] + ".mgf") 


        #print(f"mass:{row['New_PrecursorMz']},sample:{prefix}")
        # Second loop: iterate through the sorted scans in the identified group
        # print(f"mass:{row['New_PrecursorMz']}, row:{row['Modified.Sequence'], row['RT.Start'], row['RT.Stop'], row['New_PrecursorMz']}")
        for scan, (rtinseconds, (pepmass_left, pepmass_right)) in grouped_scan_map.get(filename, []):
            
            # Check if rtinseconds and New_PrecursorMz are within the specified range
            if row['RT.Start'] <= rtinseconds <= row['RT.Stop'] and pepmass_left <= row['New_PrecursorMz'] <= pepmass_right:
                related_scans.append(scan)
          
        # Concatenate the related scans into a string
        related_scans_str = ";".join(related_scans)
        related_scans_list.append(related_scans_str)
        
    # 将新生成的列添加到原始的DataFrame
    report_df['Related_Scans'] = related_scans_list
    
    return report_df


def generate_feature_csv(report_df, output_csv_path):
    
    # 初始化一个空的列表用于存储生成的 spec_group_id
    spec_group_ids = []

    # Initialize an empty DataFrame to store the new columns
    new_df = pd.DataFrame()

    # 使用正则表达式来匹配数字范围，如 "400_450"
    pattern = re.compile(r'(\d+)_(\d+)')
    
    # 遍历 report.csv 的每一行
    for index, row in report_df.iterrows():
        # 从 File.Name 列中提取 FX
        file_name = row['Run'] + ".mgf"
        FX = map_to_sample(file_name)
        # 生成 spec_group_id
        spec_group_id = f"{FX}:{index}"
        spec_group_ids.append(spec_group_id)
    
    # 将生成的 spec_group_id 列添加到 DataFrame
    new_df['spec_group_id'] = spec_group_ids

    # Create the 'm/z' column based on 'New_PrecursorMz'
    new_df['m/z'] = report_df['New_PrecursorMz']

    new_df['z'] = report_df['Precursor.Charge']
    
    # Create the 'rt_mean' column based on 'RT'
    new_df['rt_mean'] = report_df['RT']
    
    # Create the 'seq' column based on 'Modified.Sequence'
    # Replace 'C(UniMod:4)' with 'C(+57.02)'
    new_df['seq'] = report_df['Modified.Sequence'].str.replace('C(UniMod:4)', 'C(+57.02)')

    new_df['scans'] = report_df['Related_Scans']

    # Create the 'profile' column with the same length as the DataFrame but with all values set to '1:1'
    new_df['profile'] = ['1:1'] * len(report_df)

    new_df['feature area'] =  report_df['Ms1.Area']

    # Create the 'pepmass' column with the same length as the DataFrame but with all values set to empty
    new_df['pepmass'] = [''] * len(report_df)

    new_df['RTINSECONDS'] = [''] * len(report_df)

    # 删除'scans'列中空值的行
    new_df = new_df.dropna(subset=['scans'])
    
    # Save the new DataFrame to a new CSV file
    new_df.to_csv(output_csv_path, index=False)
    



# 2. Merge the precursorMz from the report-lib.csv file into the preport.csv file, these files are all generated by DIA-NN.
# # report_file_path = "/home/azureuser/biatnovo/plasma_test/report.tsv"
# # report_lib_path = "/home/azureuser/biatnovo/plasma_test/report-lib.tsv"
report_file_path = "/root/biatnovo/DeepNovo-DIA/uti/output/report.tsv"
report_lib_path = "/root/biatnovo/DeepNovo-DIA/uti/output/report-lib.tsv"
new_report_df = merge_report_and_lib(report_file_path, report_lib_path)

# 1. load map from .mgf file
# # mgf_path = "/home/azureuser/biatnovo/self_make_output/plasma/testing_plasma.spectrum.mgf"
mgf_path = "/root/biatnovo/DeepNovo-DIA/uti/uti_test.spectrum.mgf" # UTI path
# # pepmass_window = 1.0
print("Parsing MGF file...")
mgf_map = parse_mgf(mgf_path)
print("Finished parsing MGF file.")

# # 3. 建立谱图和肽段特征的联系
new_report_df = relate_scans_with_report(new_report_df, mgf_map)

# # 4. create .feature.csv
output_path = "/root/biatnovo/DeepNovo-DIA/uti/uti_test.feature.csv"
generate_feature_csv(new_report_df, output_path)
