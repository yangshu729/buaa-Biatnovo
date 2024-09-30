import os
import re

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

def convert_and_sort_mgf_files_optimized(input_dir, output_file_path):
    # Initialize a buffer list to temporarily store the converted lines for each block
    buffer = []

    # Get a list of all the .mgf files in the input directory and sort them
    mgf_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.mgf') and f in files])
    
    # Open the output file in write mode
    with open(output_file_path, 'w') as out_f:
        
        # Iterate through all the sorted files
        for filename in mgf_files:
            print(f'Begin to process file: {filename}')
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
                        match = re.search(r'File:(.*?\.raw)', title)
                        if match:
                            new_title = match.group(1)
                            buffer.append(f'TITLE={new_title}')
                            scan_prefix = map_to_sample(filename)  # 根据当前文件名生成 FX
                            match2 = re.search(r'scan=(\d+)', line)
                            if match2:
                                scans = scan_prefix + ":" + match2.group(1)
                            else:
                                print(f'Failed to find scan= in line: {line}')
                    elif line.startswith('RTINSECONDS='):
                        rtinseconds = float(line.split('=')[1])
                    elif line.startswith('PEPMASS='):
                        pepmass = line.split('=')[1].split(' ')[0]
                        buffer.append(f'PEPMASS={pepmass}')
                        buffer.append('CHARGE=0')
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

input_dir = "/root/biatnovo/train-data/ftp.peptideatlas.org/mgf-files"
convert_and_sort_mgf_files_optimized(input_dir, "/root/biatnovo/DeepNovo-DIA/uti/uti_test.spectrum.mgf")
