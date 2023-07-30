# change the old deepnovo data to new format
import csv
import re
from dataclasses import dataclass
import argparse

num_0_500 = 0
num_500_1000 = 0


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


def statistic_mgf(mgf_file_name):
    with open(mgf_file_name, "r") as fr:
        flag = False
        index = 0
        for line in fr:
            if "BEGIN ION" in line:
                flag = True
            elif not flag:
                continue
            elif line.startswith("TITLE="):
                continue
            elif line.startswith("PEPMASS="):
                continue
            elif line.startswith("CHARGE="):
                continue
            elif line.startswith("SCANS="):
                continue
            elif line.startswith("RTINSECONDS="):
                continue
            elif line.startswith("SEQ="):
                continue
            elif line.startswith("END IONS"):
                # print(index)
                # import pdb;pdb.set_trace()
                global num_0_500
                global num_500_1000
                if index < 500:
                    num_0_500 += 1
                else:
                    num_500_1000 += 1
                flag = False
                index = 0

            else:
                index += 1


if __name__ == "__main__":
    species_name = "clambacteria"

    folder_name = "/home/fzx1/cross.9high_80k.exclude_{}/".format(species_name)

    denovo_mgf_file = folder_name + "clambacteria_spectrum.mgf"

    statistic_mgf(denovo_mgf_file)
    print(num_0_500)
    print(num_500_1000)
