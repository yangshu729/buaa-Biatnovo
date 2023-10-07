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
convert_and_sort_mgf_files_optimized('/home/azureuser/biatnovo/plasm_MSConvert/',
                                      '/home/azureuser/biatnovo/self_make_output/plasma/testing_plasma.spectrum.mgf',
                                      400, 50)


# Example usage:
# convert_mgf_files_without_ctrl_m('input_mgf_files/', 'output_converted_without_ctrl_m.mgf')
