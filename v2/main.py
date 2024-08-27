import datetime
import logging
import logging.config
import os
import time
import torch
import deepnovo_config
from v2.data_reader import DeepNovoDenovoDataset, denovo_collate_func
from v2.denovo import DeepNovoAttionDenovo
from v2.model import InferenceModelWrapper
from v2.train_func import create_model, train, create_sb_model
from v2.writer import DenovoWriter

logger = logging.getLogger(__name__)
def main():
    pid = os.getpid()
    logger.info(f"pid:{pid}")
    if deepnovo_config.args.train:
        logger.info("training mode")
        train()
    elif deepnovo_config.args.search_denovo:
        logger.info("denovo mode")
        start_time = time.time()
        data_reader = DeepNovoDenovoDataset(feature_filename=deepnovo_config.denovo_input_feature_file,
                                            spectrum_filename=deepnovo_config.denovo_input_spectrum_file)
        denovo_data_loader = torch.utils.data.DataLoader(dataset=data_reader, batch_size=deepnovo_config.batch_size_predict,
                                                         shuffle=False,
                                                         num_workers=deepnovo_config.num_workers,
                                                         collate_fn=denovo_collate_func)
        denovo_worker = DeepNovoAttionDenovo(deepnovo_config.MZ_MAX,
                                             deepnovo_config.knapsack_file,
                                             beam_size=deepnovo_config.beam_size)
        
        if deepnovo_config.is_sb:
            sbatt_model, spectrum_cnn = create_sb_model(deepnovo_config.dropout_keep)
            model_wrapper = InferenceModelWrapper(sbatt_model=sbatt_model, spectrum_cnn=spectrum_cnn)
        else:
            forward_deepnovo, backward_deepnovo, spectrum_cnn = create_model(deepnovo_config.dropout_keep)
            model_wrapper = InferenceModelWrapper(forward_model=forward_deepnovo, backward_model=backward_deepnovo, spectrum_cnn=spectrum_cnn)
        writer = DenovoWriter(deepnovo_config.denovo_output_file)
        logger.info("denovo_output_file: %s", deepnovo_config.denovo_output_file)
        with torch.no_grad():
            denovo_worker.search_denovo(model_wrapper, denovo_data_loader, writer)
            # cProfile.runctx("denovo_worker.search_denovo(model_wrapper, denovo_data_loader, writer)", globals(), locals())
        logger.info(f"de novo {len(data_reader)} spectra takes {time.time() - start_time} seconds")

if __name__ == '__main__':
     # Get current date and time
    current_time = datetime.datetime.now()

    # Format the datetime string
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Create the log file name with the current date and time
    log_file_name = f"{deepnovo_config.train_dir}/biatnovo_{formatted_time}.log"
    print(f"log_file_name:{log_file_name}")
    d = {
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_name,
                'mode': 'w',
                'formatter': 'standard',
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        }
    }
    logging.config.dictConfig(d)
    main()
