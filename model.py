from tensorflow.keras.models import model_from_json
from tensorflow.python.keras.backend import set_session
import numpy as np
from tensorflow import keras
import tensorflow as tf

config = tf.ConfigProto(

    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.Session(config=config)

tf.compat.v1.keras.backend.set_session(session)

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.15
# session = tf.compat.v1.Session(config=config)
# set_session(session)


class FacialExpressionModel(object):

    EMOTIONS_LIST = ["Angry", "Disgust",
                     "Fear", "Happy",
                     "Neutral", "Sad",
                     "Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        # self.loaded_model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
        self.loaded_model._make_predict_function()

    def predict_emotion(self, img):
        with session.as_default():
            with session.graph.as_default():
                set_session(session)
                self.preds = self.loaded_model.predict(img)
                return FacialExpressionModel.EMOTIONS_LIST[np.argmax(self.preds)]
