import sys
import logging
from transformers import HfArgumentParser

from trainer import TrainArgs, Trainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    yaml_file = sys.argv[1]

    parser = HfArgumentParser((TrainArgs))

    try:
        train_args, = parser.parse_yaml_file(yaml_file=yaml_file)
    except Exception as e:
        logging.error(f"Error parsing YAML file: {e}")
        exit()

    logging.info(train_args)
    
    trainer = Trainer(train_args)
    trainer.greedy_soup()