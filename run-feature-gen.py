import logging
from utils.logger import ColoredLogger
from utils.dataset import generate_feature


logging.setLoggerClass(ColoredLogger)
logger = logging.getLogger(__name__)


generate_feature(split = 'train', block_size = 2, cell_size = 4, bin_size = 9)
logger.info('Finish feature generation for (2, 4, 9) training')

generate_feature(split = 'test', block_size = 2, cell_size = 4, bin_size = 9)
logger.info('Finish feature generation for (2, 4, 9) testing')