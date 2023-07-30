# change the old deepnovo data to new format
import csv
import re
from dataclasses import dataclass
import argparse


@dataclass
class Feature:
    spec_id: str
    mz: str
    z: str
    rt_mean: str
    seq: str
    scan: str

    def to_list(self):
        return [self.spec_id, self.mz, self.z, self.rt_mean, self.seq, self.scan, "0.0:1.0", "1.0"]


def transfer_mgf(old_mgf_file_name, output_feature_file_name, spectrum_fw=None):
    with open(old_mgf_file_name, "r") as fr:
        with open(output_feature_file_name, "w") as fw:
            writer = csv.writer(fw, delimiter=",")
            header = ["spec_group_id", "m/z", "z", "rt_mean", "seq", "scans", "profile", "feature area"]
            writer.writerow(header)
            flag = False
            for line in fr:
                if "BEGIN ION" in line:
                    flag = True
                    spectrum_fw.write(line)
                elif not flag:
                    spectrum_fw.write(line)
                elif line.startswith("TITLE="):
                    spectrum_fw.write(line)
                elif line.startswith("PEPMASS="):
                    mz = re.split("=|\r|\n", line)[1]
                    spectrum_fw.write(line)
                elif line.startswith("CHARGE="):
                    z = re.split("=|\r|\n|\+", line)[1]
                    spectrum_fw.write("CHARGE=" + z + "\n")
                elif line.startswith("SCANS="):
                    scan = re.split("=|\r|\n", line)[1]
                    spectrum_fw.write(line)
                elif line.startswith("RTINSECONDS="):
                    rt_mean = re.split("=|\r|\n", line)[1]
                    spectrum_fw.write(line)
                elif line.startswith("SEQ="):
                    seq = re.split("=|\r|\n", line)[1]
                elif line.startswith("END IONS"):
                    feature = Feature(spec_id=scan, mz=mz, z=z, rt_mean=rt_mean, seq=seq, scan=scan)
                    writer.writerow(feature.to_list())
                    flag = False
                    del scan
                    del mz
                    del z
                    del rt_mean
                    del seq
                    spectrum_fw.write(line)
                else:
                    spectrum_fw.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ==============================================================================
    # FLAGS (options) for train
    # ==============================================================================
    parser.add_argument("--data_convert", action="store_true", default=False, help="Set to True for converting data.")
    parser.add_argument("--species", type=str, default="", help="Species data")
    parser.add_argument("--folder_name", type=str, default="", help="Folder to convert new format.")
    opt = parser.parse_args()
    species_name = opt.species
    folder_name = opt.folder_name + "cross.9high_80k.exclude_{}/".format(species_name)
    # train_mgf_file = folder_name + 'cross.cat.mgf.train.repeat'
    # valid_mgf_file = folder_name + 'cross.cat.mgf.valid.repeat'
    # test_mgf_file = folder_name + 'cross.cat.mgf.test.repeat'
    denovo_mgf_file = folder_name + "peaks.db.mgf"
    # output_mgf_file = folder_name + 'spectrum.mgf'
    # output_train_feature_file = folder_name + 'features.train.csv'
    # output_valid_feature_file = folder_name + 'features.valid.csv'
    # output_test_feature_file = folder_name + 'features.test.csv'
    denovo_output_feature_file = folder_name + "test_features_10k.csv"
    denovo_spectrum_fw = open(folder_name + "test_spectrum_10k.mgf", "w")
    # transfer_mgf(train_mgf_file, output_train_feature_file, spectrum_fw=spectrum_fw)
    # transfer_mgf(valid_mgf_file, output_valid_feature_file, spectrum_fw=spectrum_fw)
    # transfer_mgf(test_mgf_file, output_test_feature_file, spectrum_fw=spectrum_fw)
    transfer_mgf(denovo_mgf_file, denovo_output_feature_file, denovo_spectrum_fw)
    denovo_spectrum_fw.close()
