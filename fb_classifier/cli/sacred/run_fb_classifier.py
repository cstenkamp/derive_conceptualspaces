#make tf shut up
import os, logging
from os.path import join, dirname, splitext, basename
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
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from derive_conceptualspace.settings import get_setting
from fb_classifier.preprocess_data import preprocess_data, create_traintest
from fb_classifier.dataset import load_data
from fb_classifier.train import TrainPipeline
# from src.fb_classifier.util.misc import get_all_debug_confs, clear_checkpoints_summary
from fb_classifier.util.misc import get_all_configs
from fb_classifier.settings import CLASSIFIER_CHECKPOINT_PATH, SUMMARY_PATH, MONGO_URI, DATA_BASE
import fb_classifier

ex = Experiment("Fachbereich_Classifier")
ex.observers.append(MongoObserver(url=MONGO_URI, db_name=os.environ["MONGO_DATABASE"]))
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.add_config(get_all_configs(as_string=False))
ex.add_config(DEBUG=get_setting("DEBUG"))
for pyfile in [join(path, name) for path, subdirs, files in os.walk(dirname(fb_classifier.__file__)) for name in files if splitext(name)[1] == ".py"]:
    ex.add_source_file(pyfile)

@ex.main
def run_experiment(_run):
    args = parse_command_line_args()
    setup_logging(args.loglevel, args.logfile)

    # if args.restart: #delete checkpoint- and summary-dir
    #     clear_checkpoints_summary()

    if args.no_continue or not os.listdir(CLASSIFIER_CHECKPOINT_PATH):
        classifier_checkpoint_path = join(CLASSIFIER_CHECKPOINT_PATH, str(_run._id))
        summary_path = join(SUMMARY_PATH, str(_run._id))
    else:
        latest_exp = max(int(i) for i in os.listdir(CLASSIFIER_CHECKPOINT_PATH))
        classifier_checkpoint_path = join(CLASSIFIER_CHECKPOINT_PATH, str(latest_exp))
        assert latest_exp == max(int(i) for i in os.listdir(SUMMARY_PATH))
        summary_path = join(SUMMARY_PATH, str(latest_exp))
        logging.warning(f"Continuing run {latest_exp}")
        #TODO check if the configs are the same and the run is not through yet!

    if get_setting("DEBUG"):
        tf.config.run_functions_eagerly(True)
        tf.data.experimental.enable_debug_mode()

    raw_data = os.path.join(DATA_BASE, "kurse-beschreibungen.csv")
    traintest = create_traintest(raw_data)
    data = preprocess_data(traintest, os.path.join(DATA_BASE, "preprocessed_dataset"), force_overwrite=False)

    pipeline = TrainPipeline(*load_data(data), ex=ex, classifier_checkpoint_path=classifier_checkpoint_path, summary_path=summary_path)

    didnt_train = False
    try:
        didnt_train = pipeline.train()
    except KeyboardInterrupt:
        pass #ensure to still save checkpoints & metrics

    if not didnt_train:
        logging.info("Adding Checkpoint and Metrics to Sacred...")
        for fname in os.listdir(pipeline.summary_path):  #used ot be pipeline.summary_writer._metadata["logdir"]._numpy().decode("UTF-8")
            ex.add_artifact(join(pipeline.summary_path, fname))
        for fname in [join(dirname(pipeline.last_save_path),i) for i in os.listdir(dirname(pipeline.last_save_path)) if splitext(i)[0] == splitext(basename(pipeline.last_save_path))[0]]+[join(dirname(pipeline.last_save_path), "checkpoint")]:
            ex.add_artifact(fname)
    else:
        pass
        #TODO abort & delete sacred run


def parse_command_line_args():
    global NON_SACRED_ARGV, SACRED_ARGV
    sys.argv = NON_SACRED_ARGV
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', dest='loglevel', default='WARNING',
                        help='log-level for logging-module. one of [DEBUG, INFO, WARNING, ERROR, CRITICAL]',)
    parser.add_argument('--logfile', dest='logfile', default='',
                        help='logfile to log to. If not set, it will be logged to standard stdout/stderr')
    # parser.add_argument('--restart', default=False, action='store_true',
    #                     help='If you want to delete checkpoint and logging and restart from scratch',)
    parser.add_argument('--no-continue', default=False, action='store_true',
                         help='If you dont want to continue from the last checkpoint (default True)',)
    parsed_args = parser.parse_args()
    sys.argv = SACRED_ARGV
    return parsed_args


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
    if "--log" in sys.argv and sys.argv.index("--log") > 0:
        # NON_SACRED_ARGV = sys.argv[:sys.argv.index("--log")] + [sys.argv[sys.argv.index("--log") + 2]]
        NON_SACRED_ARGV = sys.argv
        sys.argv = SACRED_ARGV = [sys.argv[0]]+sys.argv[sys.argv.index("--log"):sys.argv.index("--log")+2]
        #we pass the log-level to sacred but the rest we don't.
    else:
        NON_SACRED_ARGV = []
    ex.run_commandline()
    #TODO early stopping https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping