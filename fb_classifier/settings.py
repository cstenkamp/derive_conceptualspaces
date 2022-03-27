from os.path import join
import os

MONGO_URI = f"mongodb://{os.environ.get('MONGO_INITDB_ROOT_USERNAME')}:{os.environ.get('MONGO_INITDB_ROOT_PASSWORD')}@127.0.0.1/?authMechanism=SCRAM-SHA-1"

DATA_BASE = "/home/chris/Documents/UNI_neu/Masterarbeit/data_new/fb_classifier"

#ANN settings
PARAGRAPH_ENCODER = 'https://tfhub.dev/google/universal-sentence-encoder-multilingual/3'
#used to be universal-sentence-encoder-multilingual-large, but see their description @ https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3:
#   "Input text can have arbitrary length! However, model time and space complexity is $$O(n^2)$$ for input length $$n$$.
#    We recommend inputs that are approximately one sentence in length."
#see https://tfhub.dev/google/collections/universal-sentence-encoder/1 for all models
TRAIN_ENCODER = False
ENCODER_OUTDIM = 512
BATCH_SIZE = 32
ANN_EPOCHS = 12

CHECKPOINT_ALL_EPOCHS = 1
LOG_ALL_EPOCHS = 1
CHECKPOINT_LOG_ALL_TIME = '5 hours'
LABEL_NAME = 'class'
DPOINT_NAME = 'text'
DEV_MACHINE = 'chris-ThinkPad-E480'
DOMINANT_METRIC = 'test.loss'

DEBUG_SHOW_ANN_INPUT = False
DEBUG_TINY_DATASET = False
DEBUG = False

CLASSIFIER_CHECKPOINT_PATH = join(DATA_BASE, 'classifier_checkpoints')
SUMMARY_PATH = join(DATA_BASE, 'summaries')

PP_TRAIN_PERCENTAGE = 0.9
PP_DATASET = "siddata2022"

########################################################################################################################
########################################################################################################################
########################################################################################################################
# from os.path import isfile, abspath
#
# #actually make all defined directories (global vars that end in "_PATH")
# for key, val in dict(locals()).items():
#     if key.endswith('_PATH') and not isfile(val):
#         locals()[key] = abspath(val)
#         os.makedirs(locals()[key], exist_ok=True)