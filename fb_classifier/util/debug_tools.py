from functools import wraps
import tensorflow as tf
from derive_conceptualspace.settings import get_setting

def debug_tf_function(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not get_setting("DEBUG"):
            return tf.function(fn(*args, **kwargs))
        return fn(*args, **kwargs)
    return wrapped
