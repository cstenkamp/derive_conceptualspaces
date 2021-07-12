from functools import wraps
import tensorflow as tf
from src.static.settings import DEBUG

def debug_tf_function(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not DEBUG:
            return tf.function(fn(*args, **kwargs))
        return fn(*args, **kwargs)
    return wrapped
