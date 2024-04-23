import logging
import deepnovo_config
from v2.train_func import train
logger = logging.getLogger(__name__)

def main():
    if deepnovo_config.args.train:
        logger.info("training mode")
        train()