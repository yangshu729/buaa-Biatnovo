import os

import pandas as pd
import re
from collections import defaultdict


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


def read_mgf(file_path, window):
    mgf_map = {}
    with open(file_path, 'r') as f:
        rtinseconds = None
        pepmass = None
        scans = None
        for line in f:
            line = line.strip()
            if line.startswith("RTINSECONDS="):
                rtinseconds = float(line.split("=")[1])
            elif line.startswith("PEPMASS="):
                pepmass = float(line.split("=")[1])
            elif line.startswith("SCANS="):
                scans = line.split("=")[1]
            elif line == "END IONS":
                if scans is not None:
                    mgf_map[scans] = (rtinseconds, (pepmass - window, pepmass + window))
                # Reset variables for the next ion block
                rtinseconds = None
                pepmass = None
                scans = None
    return mgf_map


def process_report_xic_dir(report_xic_dir_path):
    file_ls = os.listdir(report_xic_dir_path)
    file_ls = [os.path.join(report_xic_dir_path, file) for file in file_ls]
    all_XIC_df = pd.DataFrame()
    for file in file_ls:
        xic_df = read_xic_parquet_file(file)
        all_XIC_df = pd.concat([all_XIC_df, xic_df], ignore_index=True)
    return all_XIC_df


def read_xic_parquet_file(xic_path):
    xic_df = pd.read_parquet(xic_path)
    # filter ms1 feature in .xic.parquet file
    xic_df = xic_df.loc[xic_df.feature == 'ms1']
    l = len(xic_df)
    # add file name to xic_df dataframe for condition check later
    # xic_df['file_name'] = [xic_path.split(".xic.parquet")[0]]*l
    # file_name: DIA_X2.xic.parquet
    xic_df['file_name'] = [xic_path.split("/")[-1]] * l
    return xic_df


def relate_scans_and_ms1_with_report(report_df, scan_map, xic_df, mz_begin, window_size):
    # 初始化一个空的列表用于存储与每行关联的scan字符串
    related_scans_list = []
    ms1_intensity = []

    # Create a dictionary to group scans by their 'F' prefix
    grouped_scan_map = defaultdict(list)
    for scan, values in scan_map.items():
        prefix = scan.split(":")[0]
        grouped_scan_map[prefix].append((scan, values))

    # Sort the scans within each group
    for prefix, scans in grouped_scan_map.items():
        grouped_scan_map[prefix] = sorted(scans, key=lambda x: x[1][0])

    total_rows = len(report_df)

    # for test
    # report_df = report_df.loc[:100]

    # First loop: iterate through each row in report.csv
    for index, row in report_df.iterrows():
        intensity_list = []
        related_scans = []
        row_file_name = row['File.Name']
        match = re.search(r'[^\\/:*?"<>|\r\n]+\.raw$', row_file_name)
        if match:
            row_basename = match.group(0)
        else:
            row_basename = None
        print(row_basename)
        # Print progress after every 100 rows
        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{total_rows} rows.")
        # Identify the 'F' prefix from New_PrecursorMz
        lower_bound = int(row['New_PrecursorMz']) // window_size * window_size
        prefix = f"F{(lower_bound - mz_begin) // window_size + 1}"
        # print(f"mass:{row['New_PrecursorMz']},sample:{prefix}")

        # select xic data with file_name and seq
        seq = row['Precursor.Id']
        xic_filter_df = xic_df.loc[
            (xic_df.file_name == row_basename.split('.')[0] + '.xic.parquet') & (xic_df.pr == seq)]
        # Second loop: iterate through selected xic data and check rt in order to find ms1 intensity accordingly
        for index_xic_filter, row_filter in xic_filter_df.iterrows():
            rt = row_filter['rt']
            if row['RT.Start'] <= rt <= row['RT.Stop']:
                intensity_list.append(repr(row_filter['value']))

        # for index_xic,row_xic in xic_df.iterrows():
        #     xic_file_name = row_xic['file_name']
        #     # print(xic_file_name)
        #     pr = row_xic['pr']
        #     rt = row_xic['rt']
        #     if xic_file_name.split('.')[0] == row_basename.split('.')[0] and pr==row['Precursor.Id'] and row['RT.Start'] <= rt <= row['RT.Stop']:
        #         intensity_list.append(row_xic['value'])
        intensities = ",".join(intensity_list)
        ms1_intensity.append(intensities)

        # Third loop: iterate through the sorted scans in the identified group
        for scan, (rtinseconds, (pepmass_left, pepmass_right)) in grouped_scan_map.get(prefix, []):

            # Check if rtinseconds and New_PrecursorMz are within the specified range
            if row['RT.Start'] <= rtinseconds <= row['RT.Stop'] and pepmass_left <= row[
                'New_PrecursorMz'] <= pepmass_right:
                related_scans.append(scan)

        # Concatenate the related scans into a string
        related_scans_str = ";".join(related_scans)
        related_scans_list.append(related_scans_str)

    # 将新生成的列添加到原始的DataFrame
    report_df['Related_Scans'] = related_scans_list
    report_df['MS1_Intensity'] = ms1_intensity
    return report_df


def generate_feature_csv(report_df, output_csv_path, scan_range=(400, 1000), window_size=50):
    # 初始化一个空的列表用于存储生成的 spec_group_id
    spec_group_ids = []

    # Initialize an empty DataFrame to store the new columns
    new_df = pd.DataFrame()

    # 使用正则表达式来匹配数字范围，如 "400_450"
    pattern = re.compile(r'(\d+)_(\d+)')

    # 遍历 report.csv 的每一行
    for index, row in report_df.iterrows():
        # 从 File.Name 列中提取 FX
        file_name = row['File.Name']
        match = pattern.search(file_name)

        if match:
            start, end = map(int, match.groups())

            # 计算 FX
            FX = 'F' + str((start - scan_range[0]) // window_size + 1)

            # 生成 spec_group_id
            spec_group_id = f"{FX}:{index}"
            spec_group_ids.append(spec_group_id)
        else:
            print(f"Failed to find scan window in File.Name:{file_name}")
            spec_group_ids.append(None)

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
    # new_df['profile'] = ['1:1'] * len(report_df)
    new_df['profile'] = report_df['MS1_Intensity']

    new_df['feature area'] = report_df['Ms1.Area']

    # Create the 'pepmass' column with the same length as the DataFrame but with all values set to empty
    new_df['pepmass'] = [''] * len(report_df)

    new_df['RTINSECONDS'] = [''] * len(report_df)

    # 删除'scans'列中空值的行
    new_df = new_df.dropna(subset=['scans'])

    # Save the new DataFrame to a new CSV file
    new_df.to_csv(output_csv_path, index=False)


# 1. load map from .mgf file
mgf_path = "/Users/vera/Python_Hello/BiatNovo_DataProcess/mgf_data/testing_oc.spectrum.mgf"
pepmass_window = 1.0
mgf_map = read_mgf(mgf_path, pepmass_window)
print(len(mgf_map))

# 2. Merge the precursorMz from the report-lib.csv file into the preport.csv file, these files are all generated by DIA-NN.
report_file_path = "/Users/vera/Python_Hello/BiatNovo_DataProcess/20240725_6test/report.tsv"
report_lib_path = "/Users/vera/Python_Hello/BiatNovo_DataProcess/20240725_6test/report-lib.tsv"
new_report_df = merge_report_and_lib(report_file_path, report_lib_path)

# 3. load all the .xic.parquet files from report_xic dir
xic_dir_path = r'/Users/vera/Python_Hello/BiatNovo_DataProcess/20240725_6test/report_xic' # report_xic directory path
XIC_df = process_report_xic_dir(xic_dir_path)

# 4. 建立谱图和肽段特征的联系
new_report_df = relate_scans_and_ms1_with_report(new_report_df, mgf_map, XIC_df,400, 50)

# 5. create .feature.csv
output_path = "/Users/vera/Python_Hello/BiatNovo_DataProcess/20240725_6test/testing_oc_with_ms1.feature.csv"
generate_feature_csv(new_report_df, output_path)
