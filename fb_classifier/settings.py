from os.path import join
from src.static.settings import DATA_BASE

#ANN settings
PARAGRAPH_ENCODER = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3'
#see https://tfhub.dev/google/collections/universal-sentence-encoder/1 for all models
TRAIN_ENCODER = False
ENCODER_OUTDIM = 512
BATCH_SIZE = 32
ANN_EPOCHS = 10

CHECKPOINT_ALL_EPOCHS = 1
LOG_ALL_EPOCHS = 1
CHECKPOINT_LOG_ALL_TIME = '5 hours'
LABEL_NAME = 'fachbereich'
DPOINT_NAME = 'coursename'
DEV_MACHINE = 'chris-ThinkPad-E480'
DOMINANT_METRIC = 'test.loss'

DEBUG_SHOW_ANN_INPUT = False
DEBUG_TINY_DATASET = False

CLASSIFIER_CHECKPOINT_PATH = join(DATA_BASE, 'classifier_checkpoints')
SUMMARY_PATH = join(DATA_BASE, 'summaries')

PP_TRAIN_PERCENTAGE = 0.9