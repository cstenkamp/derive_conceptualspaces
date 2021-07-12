'''
When running normally outside of PyCharm:
    setup conda-environment with python >= 3.7, pip install -r requirements inside that
    in the base-dir of this project, run PYTHONPATH="./src:$PYTHONPATH" python scripts/start_doc2vec.py [--log INFO]
When running normally inside PyCharm:
    setup conda-environment with python >= 3.7, pip install -r requirements inside that
    select this environment as interpreter
    mark "src" directory as "sources root"
When running inside PyCharm with SSH-Interpreter:
    ssh to the place you're running on, setup conda-env as stated above
    figure out path of that interpreter and set that as interpreter in the scripts/condaactivate_python script
    select this script as your remote-interpreter (not the conda/envs/.../python!)
    add LD_LIBRARY_PATH to the environment-vars for your interpreter-run-config (see scripts/list_environment.py)
    specify the correct working-directory for your interpreter-run-config
    mark "src" directory as "sources root"
'''
"""
TODO:
-Using BERT tokenization instead of NLTK? (part of huggingface: https://huggingface.co/transformers/main_classes/tokenizer.html)

INFO FOR doc2vec:
-Gensim Model: https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py
-TF Model (& comparison to gensim model): https://stackoverflow.com/questions/39843584/gensim-doc2vec-vs-tensorflow-doc2vec
-Best TF Implementation: https://github.com/PacktPublishing/TensorFlow-Machine-Learning-Cookbook/blob/master/Chapter%2007/doc2vec.py
-Only pretrained doc2vec Model I found was for gensim: https://stackoverflow.com/questions/51132848/is-there-pre-trained-doc2vec-model
    -stuck at https://github.com/jhlau/doc2vec/issues/23
-HowTo train gensim Model on Wikipedia-Dataset: https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-wikipedia.ipynb
-HowTo train tf Model on Wikipedia-Dataset: https://aihub.cloud.google.com/p/products%2F882189a2-dff6-4345-90eb-67731d8c82f1
-Pretrained word2vec: https://wikipedia2vec.github.io/wikipedia2vec/pretrained/
-Pretrained doc2vec: https://tfhub.dev/google/universal-sentence-encoder/4 !!

Questions:
-If doc2vec is pretrained on wikipedia, is it trained on the whole article or on individual sentences?


...soooo...:
-Use https://tfhub.dev/google/collections/universal-sentence-encoder/1.
-Make eval-script what the inter-class variance and intra-class-variance is
-Try to optimize a bit (incorporate word-vectors, incorporate tf-idf) -> nen kleinen kopf machen der die word-vectors nach tf-idf wertet?
-maybe run LDA (https://en.wikipedia.org/wiki/Linear_discriminant_analysis) on the paragraph-vectors to optimize for DDCs?

TODO
* use sacred!!!
"""

#make tf shut up
import os, logging
loglevel = "ERROR"
numeric_level = getattr(logging, loglevel.upper(), None)
tf_log_translator = {"INFO": "0", "DEBUG": "0", "WARNING": "2", "ERROR": "2",
                     "CRITICAL": "3"}  # https://stackoverflow.com/a/42121886/5122790
os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_log_translator[loglevel.upper()]
os.environ['LOG_LEVEL'] = str(numeric_level)
import tensorflow as tf
tf.get_logger().setLevel(loglevel.upper())
#/make tf shut up

import tensorflow as tf
import argparse
import logging
import os, site, socket, sys

from src.static.classifier_config import DEV_MACHINE
from src.static.settings import SID_DATA_BASE, DEBUG
from src.fb_classifier.preprocess_data import preprocess_data, create_traintest
from src.fb_classifier.dataset import load_data
from src.fb_classifier.train import TrainPipeline
from src.fb_classifier.util.misc import get_all_debug_confs, clear_checkpoints_summary

#TODO do this with sacred

def main():
    args = parse_command_line_args()
    setup_logging(args.loglevel, args.logfile)

    if socket.gethostname() != DEV_MACHINE or os.environ.get('JETBRAINS_REMOTE_RUN'):
        #remote-running: see https://stackoverflow.com/questions/56098247/setting-up-a-pycharm-remote-conda-interpreter and https://youtrack.jetbrains.com/issue/PY-35978
        logging.info(f'You are on machine: {socket.gethostname()}')
        logging.info(f'Your Python executable: {sys.executable}')
        logging.info(f'Your LD_LIBRARY_PATH: {os.environ.get("LD_LIBRARY_PATH", "")}')
        assert os.environ.get('CONDA_PREFIX'), "Your environment doesn't have a correct active conda-env and you are not on the dev-machine!"
        assert os.environ.get('LD_LIBRARY_PATH'), "Your environment doesn't know LD_LIBRARY_PATH, you won't find CuDNN and thus cannot use GPUs and are not on the dev-machine!"
        assert len([i for i in os.listdir(site.getsitepackages()[0]) if 'tensorflow_gpu' in i]) > 0, "Tensorflow_GPU cannot be found and you are not on the dev-machine!"
        assert not any(get_all_debug_confs().values()), "You still have DEBUG-Configs active and are not on the dev-machine!"
        assert len(tf.config.experimental.list_physical_devices('GPU')) > 0, "You don't have any GPUs and are not on the dev-machine!"
        # for GPU, see https://www.tensorflow.org/guide/gpu
        # tf.debugging.set_log_device_placement(True)

    if args.restart: #delete checkpoint- and summary-dir
        clear_checkpoints_summary()

    if DEBUG:
        tf.config.run_functions_eagerly(True)


    raw_data = os.path.join(SID_DATA_BASE, "kurse-beschreibungen.csv")
    traintest = create_traintest(raw_data)
    data = preprocess_data(traintest, os.path.join(SID_DATA_BASE, "preprocessed_dataset"), force_overwrite=False)

    pipeline = TrainPipeline(*load_data(data))
    pipeline.train()


def parse_command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='loglevel', default='WARNING',
                        help='log-level for logging-module. one of [DEBUG, INFO, WARNING, ERROR, CRITICAL]',)
    parser.add_argument('--logfile', dest='logfile', default='',
                        help='logfile to log to. If not set, it will be logged to standard stdout/stderr')
    parser.add_argument('--restart', default=False, action='store_true',
                        help='If you want to delete checkpoint and logging and restart from scratch',)
    return parser.parse_args()


def setup_logging(loglevel='WARNING', logfile=None):
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    kwargs = {'level': numeric_level, 'format': '%(asctime)s %(levelname)-8s %(message)s',
              'datefmt': '%Y-%m-%d %H:%M:%S', 'filemode': 'w'}
    if logfile:
        kwargs['filename'] = logfile
    logging.basicConfig(**kwargs)


if __name__ == '__main__':
    main()
    #TODO early stopping https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping