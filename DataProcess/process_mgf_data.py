import csv
import os
import re

def map_to_sample(title, lower_bound, window_size):
    # 使用正则表达式来匹配数字范围，如 "400_450"
    pattern = re.compile(r'(\d+)_(\d+)')
    match = pattern.search(title)
    if match:
            start, end = map(int, match.groups())
            # 计算 FX
            FX = 'F' + str((start - lower_bound) // window_size + 1)
            return FX
    print(f'Failed to find window range in title:{title}')
    return None


file_mapfile_names = [
        "DIA_A1.mgf", "DIA_A2.mgf", "DIA_A3.mgf", "DIA_A4.mgf", "DIA_A7.mgf", 
        "DIA_A8.mgf", "DIA_A9.mgf", "DIA_A10.mgf", "DIA_A11.mgf", "DIA_B1.mgf", 
        "DIA_B2.mgf", "DIA_B3.mgf", "DIA_B4.mgf", "DIA_B5.mgf", "DIA_B6.mgf", 
        "DIA_B7.mgf", "DIA_B8.mgf", "DIA_B9.mgf", "DIA_B10.mgf", "DIA_B11.mgf", 
        "DIA_B12.mgf", "DIA_C1.mgf", "DIA_C3.mgf", "DIA_C4.mgf", "DIA_C6.mgf", 
        "DIA_C7.mgf", "DIA_C9.mgf", "DIA_C11.mgf", "DIA_D3.mgf", "DIA_D4.mgf", 
        "DIA_D5.mgf", "DIA_D6.mgf", "DIA_D7.mgf", "DIA_D8.mgf", "DIA_D10.mgf", 
        "DIA_D12.mgf", "DIA_E1.mgf", "DIA_E2.mgf", "DIA_E3.mgf", "DIA_E4.mgf", 
        "DIA_E7.mgf", "DIA_E8.mgf", "DIA_E9.mgf", "DIA_E10.mgf", "DIA_E12.mgf", 
        "DIA_F2.mgf", "DIA_F3.mgf", "DIA_F4.mgf", "DIA_F5.mgf", "DIA_F7.mgf", 
        "DIA_F8.mgf", "DIA_F12.mgf", "DIA_G3.mgf", "DIA_G4.mgf", "DIA_G5.mgf", 
        "DIA_G6.mgf", "DIA_G7.mgf", "DIA_G8.mgf", "DIA_G9.mgf", "DIA_G11.mgf", 
        "DIA_G12.mgf", "DIA_H1.mgf", "DIA_H2.mgf", "DIA_H3.mgf"
    ]

file_map = {file_name: f"F{i+1}" for i, file_name in enumerate(file_mapfile_names)}

def convert_feature_file(origin_file, output_file):
    # 读取原始CSV文件
    with open(origin_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)

    # 处理每行数据
    new_rows = []
    for index, row in enumerate(rows):
        sample_id = file_map.get(row['spec_group_id'] + '.mgf' , row['spec_group_id'])
        new_spec_group_id = sample_id + ":" + str(index + 1)
        # 添加 profile 和 feature area 的值，这里暂时用空字符串代替
        new_seq = row['seq'].replace('M+15.995', 'M(+15.99)')
        new_scans = ";".join([sample_id + ":" + scan  for scan in row['scans'].split(';')])
        temp = "0:0"
        new_profile = ";".join([temp] * len(row['scans'].split(';')))

        row.update({
            'spec_group_id': new_spec_group_id,
            'seq': new_seq,
            'scans': new_scans,
            'profile': new_profile,  # 根据需要填充
            'feature area': 0  # 根据需要填充
        })
        new_rows.append(row)

    # 写入新的CSV文件
    with open(output_file, 'w', newline='') as new_file:
        fieldnames = list(rows[0].keys()) 
        writer = csv.DictWriter(new_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(new_rows)


def merge_mgf_files(input_dir, output_file_path):

    # 创建映射
    
    # Open the output file in write mode
    with open(output_file_path, 'w') as out_f:
        # 按照file_names中的顺序遍历文件
        for file_name in file_mapfile_names:
            file_path = os.path.join(input_dir, file_name)
            with open(os.path.join(input_dir, file_path), 'r') as in_f:
                    # Initialize variables to store information for each ion block
                    title = ''
                    pepmass = ''
                    rtinseconds = ''
                    scans = ''
                    scan_prefix = file_map[file_name]
                    print(file_path, scan_prefix, flush=True)
                    
                    for line in in_f:
                        line = line.strip()
                        if line.startswith('BEGIN IONS'):
                            # Reset buffer and variables for a new ion block
                            buffer = []
                            title = ''
                            pepmass = ''
                            rtinseconds = ''
                            charge = '0'
                            scans = ''
                            buffer.append('BEGIN IONS')
                        elif line.startswith('TITLE='):
                            title = line.split('=')[1]
                            # Extract new title and scan number using regular expression
                            match = re.search(r'File:(.*?\.raw)', title)
                            if match:
                                new_title = match.group(1)
                                buffer.append(f'TITLE={new_title}')
                                match2 = re.search(r'scan=(\d+)', line)
                                if match2:
                                    scans = scan_prefix + ":" + match2.group(1)                                                        
                                else:
                                    print(f'Failed to find scan= in line:{line}')
                        elif line.startswith('RTINSECONDS='):
                            rtinseconds = float(line.split('=')[1])
                        elif line.startswith('PEPMASS='):
                            pepmass = line.split('=')[1].split(' ')[0]
                            buffer.append(f'PEPMASS={pepmass}')
                            buffer.append(f'CHARGE=0')
                            buffer.append(f'SCANS={scans}')
                            buffer.append(f'RTINSECONDS={rtinseconds / 60}')
                        elif line.startswith('END IONS'):
                            # Append remaining fields and write the buffer to the output file
                            buffer.append('END IONS')
                            out_f.write('\n'.join(buffer))
                            out_f.write('\n')
                        else:
                            # Collect data points
                            buffer.append(line)

           


def convert_and_sort_mgf_files_optimized(input_dir, output_file_path, lower_bound, window_size):
    # Initialize a buffer list to temporarily store the converted lines for each block
    buffer = []

    # Get a list of all the .mgf files in the input directory and sort them
    mgf_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mgf')])
    
    # Open the output file in write mode
    with open(output_file_path, 'w') as out_f:
        
        # Iterate through all the sorted files
        for filename in mgf_files:
            print(f'begin to process file:{filename}')
            with open(os.path.join(input_dir, filename), 'r') as in_f:
                # Initialize variables to store information for each ion block
                title = ''
                pepmass = ''
                rtinseconds = ''
                scans = ''
                
                for line in in_f:
                    line = line.strip()
                    if line.startswith('BEGIN IONS'):
                        # Reset buffer and variables for a new ion block
                        buffer = []
                        title = ''
                        pepmass = ''
                        rtinseconds = ''
                        charge = '0'
                        scans = ''
                        buffer.append('BEGIN IONS')
                    elif line.startswith('TITLE='):
                        title = line.split('=')[1]
                        # Extract new title and scan number using regular expression
                        match = re.search(r'File:"(.*?\.raw)"', title)
                        if match:
                            new_title = match.group(1)
                            buffer.append(f'TITLE={new_title}')
                            scan_prefix = map_to_sample(new_title, lower_bound, window_size)
                            match2 = re.search(r'scan=(\d+)', line)
                            if match2:
                                scans = scan_prefix + ":" + match2.group(1)                                                        
                            else:
                                print(f'Failed to find scan= in line:{line}')
                    elif line.startswith('RTINSECONDS='):
                        rtinseconds = float(line.split('=')[1])
                    elif line.startswith('PEPMASS='):
                        pepmass = line.split('=')[1].split(' ')[0]
                        buffer.append(f'PEPMASS={pepmass}')
                        buffer.append(f'CHARGE=0')
                        buffer.append(f'SCANS={scans}')
                        buffer.append(f'RTINSECONDS={rtinseconds / 60}')
                    elif line.startswith('END IONS'):
                        # Append remaining fields and write the buffer to the output file
                        buffer.append('END IONS')
                        out_f.write('\n'.join(buffer))
                        out_f.write('\n')
                    else:
                        # Collect data points
                        buffer.append(line)
            

# Example usage:
# convert_and_sort_mgf_files_optimized('/home/azureuser/biatnovo/plasm_MSConvert/',
#                                       '/home/azureuser/biatnovo/self_make_output/plasma/testing_plasma.spectrum.mgf',
#                                       400, 50)


# Example usage:
# convert_mgf_files_without_ctrl_m('input_mgf_files/', 'output_converted_without_ctrl_m.mgf')
# merge_mgf_files('/home/azureuser/biatnovo/train-data/ftp.peptideatlas.org/mgf-files/', 
#                 '/home/azureuser/biatnovo/train-data/ftp.peptideatlas.org/biatNovo/training.spectrum.mgf')


convert_feature_file('/home/azureuser/diann/output/report.feature.csv',
                     '/home/azureuser/biatnovo/train-data/ftp.peptideatlas.org/biatNovo/training.feature.csv')
