from functools import partial
from os.path import dirname, join

import tensorflow as tf


from fb_classifier.settings import BATCH_SIZE, LABEL_NAME, DPOINT_NAME, DEBUG_TINY_DATASET


# https://www.tensorflow.org/guide/data, https://www.tensorflow.org/tutorials/load_data/csv, https://www.tensorflow.org/api_docs/python/tf/data/Dataset,
# https://www.tensorflow.org/tutorials/load_data/text, https://www.tensorflow.org/tutorials/load_data/pandas_dataframe
# https://www.tensorflow.org/guide/data#consuming_csv_data


def load_data(paths: dict):
    dset = {key: tf.data.experimental.make_csv_dataset(val,
                                                       batch_size=BATCH_SIZE,
                                                       shuffle=(key=='train'),
                                                       shuffle_buffer_size=100000,
                                                       column_names=[DPOINT_NAME, LABEL_NAME],
                                                       num_epochs=1) #https://github.com/tensorflow/tensorflow/issues/23785
            for key, val in paths.items()}
    # now dset is longer than the #rows in the csv, wtf!! https://www.tensorflow.org/api_docs/python/tf/data/experimental/make_csv_dataset

    # dset['train'] = dset['train'].shuffle(buffer_size=1000).batch(BATCH_SIZE, drop_remainder=True) #no repeating, bc custom statistics after epoch -> reset
    #Design decision: The encoder is part of the network, not part of the preprocessing -> dataset just creates the variable-length-sentences
    # (if not, padding could be relevant - https://www.tensorflow.org/guide/data#batching_tensors_with_padding)
    # Preprocessing: https://www.tensorflow.org/guide/data#preprocessing_data
    if DEBUG_TINY_DATASET:
        dset = {key: val.take(10) for key, val in dset.items()}
    with open(join(dirname(paths["train"]), "meta"), "r") as rfile:
        n_classes = [int(i[len("classes: "):]) for i in rfile.readlines() if i.startswith("classes: ")][0]
    return dset, n_classes


def get_dset_len(dset):
    try:
        return dset._count.numpy()
    except AttributeError:
        if dset.cardinality().numpy() not in [tf.data.experimental.UNKNOWN_CARDINALITY, tf.data.experimental.INFINITE_CARDINALITY]:
            return tf.data.experimental.cardinality(dset).numpy()
        num_elements = 0
        for _ in dset:
            num_elements += 1
        return num_elements
