from keras.layers.core import Activation, Dense, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential,load_model
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import collections
from collections import defaultdict
import numpy as np
import sys
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

input_fld = 'keras模型导出/'
output_fld = input_fld + 'tensorflow_freeze_model/'
output_graph_name = 'lstm1.pb'

K.set_learning_phase(0)
net_model = load_model(input_fld + "LSTM-Station46.h5",compile=False)

print('input is :', net_model.input.name)
print ('output is:', net_model.output.name)

sess = K.get_session()

frozen_graph = freeze_session(K.get_session(), output_names=[net_model.output.op.name])
from tensorflow.python.framework import graph_io

graph_io.write_graph(frozen_graph, output_fld, output_graph_name,as_text=False)


print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))

from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

version = "2/"
export_dir = input_fld + 'saved_model'
graph_pb = output_fld + output_graph_name

builder = tf.saved_model.builder.SavedModelBuilder(export_dir + version)

with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    # print(f.read())
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()
    inp = g.get_tensor_by_name(net_model.input.name)
    out = g.get_tensor_by_name(net_model.output.name)

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"in": inp}, {"out": out})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

builder.save()