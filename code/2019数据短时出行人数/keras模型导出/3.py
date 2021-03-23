# tensorflow == 1.13.1
import tensorflow as tf
from keras_gcn import *

def load_keras_model(model_path, weights_path):
    fr = open(model_path, "r")
    model_json = fr.read()
    fr.close()
    model = tf.keras.models.model_from_json(model_json, custom_objects={"tf":tf})
    model.load_weights(weights_path)
    return model

model_export_dir = "./model/1"
model = tf.keras.models.load_model("{}-minute forecast/models/Station{}.h5".format(15,46),custom_objects={"GraphConv":GraphConv},compile=False)
# model = load_keras_model("model.json", "weights.h5")
name_to_inputs = {i.name.split(":")[0]:i for i in model.inputs}
name_to_outputs = {i.name:i for i in model.outputs}
print(name_to_inputs)
print(name_to_outputs)
tf.saved_model.simple_save(tf.keras.backend.get_session(),
                           model_export_dir,
                           inputs=name_to_inputs,
                           outputs=name_to_outputs)