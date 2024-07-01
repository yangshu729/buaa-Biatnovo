from dataclasses import dataclass
from data_reader import DIAFeature
import deepnovo_config
import logging

logger = logging.getLogger(__name__)


@dataclass
class BeamSearchedSequence:
    sequence: list  # list of aa id
    position_score: list
    score: float  # average by length score


class DenovoWriter(object):
    def __init__(self, denovo_output_file):
        self.log_file = denovo_output_file
        
        header_list = ["feature_id",
                       "feature_area",
                       "predicted_sequence",
                       "predicted_score",
                       "predicted_position_score",
                       "precursor_mz",
                       "precursor_charge",
                       "protein_access_id",
                       "scan_list_middle",
                       "scan_list_original",
                       "predicted_score_max"]
        header_row = "\t".join(header_list)
        with open(self.log_file, 'a') as self.output_handle:
            print(header_row, file=self.output_handle, end='\n')

    def close(self):
        self.output_handle.close()

    def write(self, dia_feature: DIAFeature, searched_sequence: BeamSearchedSequence):
        """
        keep the output in the same format with the tensorflow version
        :param dia_feature:
        :param searched_sequence:
        :return:
        """
        feature_id = dia_feature.feature_id
        feature_area = str(dia_feature.feature_area)
        precursor_mz = str(dia_feature.precursor_mz)
        precursor_charge = str(dia_feature.precursor_charge)
        scan_list_middle = ','.join(dia_feature.scan_list_middle)
        scan_list_original = ','.join(dia_feature.scan_list)
        if searched_sequence.sequence:
            predicted_sequence = ','.join([deepnovo_config.vocab_reverse[aa_id] for
                                           aa_id in searched_sequence.sequence])
            predicted_score = "{:.2f}".format(searched_sequence.score)
            predicted_score_max = predicted_score
            predicted_position_score = ','.join(['{0:.2f}'.format(x) for x in searched_sequence.position_score])
            protein_access_id = 'DENOVO'
        else:
            predicted_sequence = ""
            predicted_score = ""
            predicted_score_max = ""
            predicted_position_score = ""
            protein_access_id = ""
        predicted_row = "\t".join([feature_id,
                                   feature_area,
                                   predicted_sequence,
                                   predicted_score,
                                   predicted_position_score,
                                   precursor_mz,
                                   precursor_charge,
                                   protein_access_id,
                                   scan_list_middle,
                                   scan_list_original,
                                   predicted_score_max])
        with open(self.log_file, 'a') as output_handle:
            print(predicted_row, file=output_handle, end="\n")

    def __del__(self):
        self.close()
