"""
This script handles the training process.
"""

import argparse

from DataProcessing import deepnovo_worker_test


__author__ = "Si-yu Wu"


def main():
    parser = argparse.ArgumentParser()
    # ==============================================================================
    # FLAGS (options) for test accuray
    # ==============================================================================
    parser.add_argument("--test_accuracy", action="store_true", default=False, help="Set to True to test.")
    parser.add_argument(
        "--target_file",
        type=str,
        default="target_peptide.csv",
        help="Target file to calculate the prediction accuracy.",
    )
    parser.add_argument(
        "--predict_file", type=str, default="result.txt", help="Predict file to calculate the prediction accuracy."
    )
    parser.add_argument("--accuracy_file", type=str, default="accuracy.txt", help="Accuracy file.")
    parser.add_argument("--denovo_only_file", type=str, default="denovo_only.txt", help="Deepnovo_only file.")

    opt = parser.parse_args()

    """
    Make Predict Dir
    """
    # if not os.path.exists(opt.predict_dir):
    #     os.makedirs(opt.predict_dir)
    print(opt)

    """
    test accuracy
    """
    worker_test = deepnovo_worker_test.WorkerTest(opt)
    worker_test.test_accuracy()
    #worker_test.test_accuracy_position_bleu()
    #worker_test.test_accuracy_smith_waterman()


if __name__ == "__main__":
    main()
