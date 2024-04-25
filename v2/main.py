import deepnovo_config
from v2.train_func import train
from logger_config import setup_logging
logger = setup_logging()

def main():
    if deepnovo_config.args.train:
        logger.info("training mode")
        train()

if __name__ == "__main__":
    main()