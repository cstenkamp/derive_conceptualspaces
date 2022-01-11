import tensorflow_hub as hub
import tensorflow_text #necessary to use the model from tf-hub!
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

from fb_classifier.settings import PARAGRAPH_ENCODER, ENCODER_OUTDIM, TRAIN_ENCODER

# embed = hub.KerasLayer(PARAGRAPH_ENCODER)
# embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
# print(embeddings.shape, embeddings.dtype)


class FB_Classifier(Model):
    """Classifier that uses sentence embeddings to classify sentence -> DDC"""
    def __init__(self, input_dim=(1,None), output_dim=None):
        super(self.__class__, self).__init__()
        self.embed = hub.KerasLayer(PARAGRAPH_ENCODER, input_shape=(1,), trainable=TRAIN_ENCODER)
        # self.conv1 = Conv2D(32, 3, activation='relu')
        # self.flatten = Flatten()
        #TODO: den dense part possibly as layer (https://www.tensorflow.org/guide/keras/custom_layers_and_models#the_model_class)
        self.d1 = Dense(ENCODER_OUTDIM, activation='relu')
        self.d2 = Dense(output_dim)

    def __call__(self, inputs, training=True):
        x = self.embed(inputs)  #input shape: (32,)
        x = self.d1(x)          #x shape: (32, 512)
        return self.d2(x)       #x shape: (32, 512)

    def call(self, inputs, training=True):
        return self.__call__(inputs, training=training)

    #TODO: custom loss - see https://www.tensorflow.org/guide/keras/custom_layers_and_models#the_add_loss_method

    def summary(self):
        self.build([]) #inputs.shape == (32,)
        self.d1.build(ENCODER_OUTDIM)
        self.d2.build()
        self.summary()
