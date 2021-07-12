import tensorflow as tf
import logging
from datetime import datetime
import pandas as pd
from tqdm import tqdm

from src.static.classifier_config import ANN_EPOCHS, CLASSIFIER_CHECKPOINT_PATH, CHECKPOINT_ALL_EPOCHS, LABEL_NAME, DPOINT_NAME, \
    DEBUG_SHOW_ANN_INPUT, SUMMARY_PATH, LOG_ALL_EPOCHS, CHECKPOINT_LOG_ALL_TIME, DOMINANT_METRIC
from src.fb_classifier.model import FB_Classifier
from src.fb_classifier.util.debug_tools import debug_tf_function
from src.fb_classifier.dataset import get_dset_len
from src.fb_classifier.util.misc import check_config, get_git_revision_short_hash, get_ann_configs

METRICS_DISPLAYSTYLE = {
    'loss': lambda i: f'{i:.3f}',
    'accuracy': lambda i: f'{i*100:.2f}%'
}

flatten_dict = lambda data: dict((key,d[key]) for d in data for key in d)

def display_metrics(metrics):
    return ' || '.join([
        f'{setname.capitalize()}: '
        + ', '.join([ f'{metricname.capitalize()}: {METRICS_DISPLAYSTYLE[metricname](metric.result())}'
                           for metricname, metric in setmetrics.items()
                    ])
        for setname, setmetrics in metrics.items()
    ])


class TrainPipeline():
    '''See https://www.tensorflow.org/tutorials/quickstart/advanced and https://www.tensorflow.org/guide/effective_tf2'''

    def __init__(self, dataset, n_classes):
        self.dataset = dataset
        self.model = FB_Classifier(output_dim=n_classes)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #TODO from_logits? yes? no?
        self.optimizer = tf.keras.optimizers.Adam()
        self.last_checkpoint_at = datetime.now()
        self.last_logging_at = datetime.now()

        self.metrics = {
            'train': {
                'loss': tf.keras.metrics.Mean(name='train_loss', dtype=tf.float32),
                'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy', dtype=tf.float32)
            },
            'test': {
                'loss': tf.keras.metrics.Mean(name='test_loss', dtype=tf.float32),
                'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy', dtype=tf.float32)
            }
        }
        self.summary_writer = tf.summary.create_file_writer(SUMMARY_PATH)

    def save_modelinfo(self):
        dset, metric = DOMINANT_METRIC.split('.')
        dominant_metric = self.metrics[dset][metric].result().numpy()
        #timestamp, git-commit, some settings, dominant_metric, ANN-structure
        print()


    def train(self):
        # train_iterator = iter(self.dataset['train'])
        check_config(CLASSIFIER_CHECKPOINT_PATH)
        self._ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=self.optimizer, net=self.model) #, iterator=train_iterator (see to-do in train_epoch)
        self._manager = tf.train.CheckpointManager(self._ckpt, CLASSIFIER_CHECKPOINT_PATH, max_to_keep=3)
        if self._manager.latest_checkpoint:
            self._ckpt.restore(self._manager.latest_checkpoint)
            logging.info(f'Restored from {self._manager.latest_checkpoint} ({self._ckpt.step.numpy()} steps already done)')
        else:
            logging.info("Initializing from scratch.")
        if self._ckpt.step.numpy() >= ANN_EPOCHS:
            logging.error("The checkpoint already has as many epochs as necessary!")
            self.epoch('test')
            logging.info(f'Metrics: {display_metrics(self.metrics)}')
            return

        for epoch in range(self._ckpt.step.numpy(), ANN_EPOCHS):
            self.epoch('train', size=get_dset_len(self.dataset['train']))
            self.epoch('test')
            logging.info(f'Epoch {str(epoch+1).rjust(3)} - Metrics: {display_metrics(self.metrics)}')
            self._ckpt.step.assign_add(1)
            self.checkpoint_summary()
            self.save_modelinfo()
        self.checkpoint_summary(force=True)


    def checkpoint_summary(self, in_epoch=False, force=False):
        if ((not in_epoch and int(self._ckpt.step) > 0 and int(self._ckpt.step) % CHECKPOINT_ALL_EPOCHS == 0) or
                (datetime.now() - self.last_checkpoint_at > pd.to_timedelta(CHECKPOINT_LOG_ALL_TIME)) or force):
            save_path = self._manager.save()
            logging.debug(f"Saved checkpoint for step {int(self._ckpt.step)} at {save_path}")
            self.last_checkpoint_at = datetime.now()
        if ((not in_epoch and tf.equal(self.optimizer.iterations % LOG_ALL_EPOCHS, 0)) or
                (datetime.now() - self.last_logging_at > pd.to_timedelta(CHECKPOINT_LOG_ALL_TIME)) or force):
            with self.summary_writer.as_default():
                for name, value in flatten_dict([{f'{setname}-{metricname}': metric.result() for metricname, metric in setmetrics.items()} for setname, setmetrics in self.metrics.items()]).items():
                    tf.summary.scalar(name, value, step=self.optimizer.iterations)
                tf.summary.scalar('epoch', self._ckpt.step, step=self.optimizer.iterations)
            self.last_logging_at = datetime.now()


    @debug_tf_function
    def epoch(self, setname='train', size=None):
        for metric in self.metrics[setname].values(): # Reset the metrics at the start of the next epoch
            metric.reset_states()

        iterator = enumerate(self.dataset[setname])
        if size: #TODO das vielleicht nur wenn debugging True? ich bezweifel dass tf das elegantieren kann
            iterator = tqdm(enumerate(self.dataset[setname]), total=size)
        for num, batch in iterator:
        #TODO feels weird not to iterate over train_iterator, see https://www.tensorflow.org/guide/data#iterator_checkpointing
        #however this being already an iterator cannot reset after an epoch, so I'd need to call the Checkpoint-constructor every epoch??
            dpoints, labels = batch[DPOINT_NAME], batch[LABEL_NAME]
            if DEBUG_SHOW_ANN_INPUT:
                print(f'  Batch #{num+1}: \n'+'\n'.join([f'   {i[0]}: {i[1].decode("UTF-8")}' for i in list(zip(labels.numpy(), dpoints.numpy()))])+'\n\n')
            if setname == 'train':
                self.train_step(dpoints, labels)
            else:
                self.test_step(dpoints, labels)


    @debug_tf_function
    def train_step(self, dpoints, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(dpoints, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.metrics['train']['loss'].update_state(loss)
        self.metrics['train']['accuracy'].update_state(labels, predictions)
        self.checkpoint_summary(in_epoch=True)


    @debug_tf_function
    def test_step(self, dpoints, labels):
        predictions = self.model(dpoints, training=False)
        t_loss = self.loss_object(labels, predictions)
        self.metrics['test']['loss'].update_state(t_loss)
        self.metrics['test']['accuracy'].update_state(labels, predictions)
