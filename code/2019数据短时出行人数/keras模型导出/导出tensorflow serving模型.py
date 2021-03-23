from keras.models import Sequential,load_model
import numpy as np
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K
from keras_gcn import *

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

output_graph_name = 'tensor_model.pb'

output_fld = 'keras模型导出/tensorflow_model/'
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)

# 返回训练模式/测试模式的flag，该flag是一个用以传入Keras模型的标记，以决定当前模型执行于训练模式下还是测试模式下.
# To make use of the learning phase, simply pass the value "1" (training mode) or "0" (test mode) to feed_dict
K.set_learning_phase(0)
net_model = load_model("{}-minute forecast/models/Station{}.h5".format(15,46),custom_objects={"GraphConv":GraphConv},compile=False)


# print('input is :', net_model.input.name)
print ('output is:', net_model.output.name)

sess = K.get_session()

frozen_graph = freeze_session(K.get_session(), output_names=[net_model.output.op.name])
from tensorflow.python.framework import graph_io

graph_io.write_graph(frozen_graph, output_fld, output_graph_name)


print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))


from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

export_dir = 'keras模型导出/saved_model'
graph_pb = 'keras模型导出/tensorflow_model/tensor_model.pb'

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.gfile.FastGFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    print("***********************")

print("================================")
sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    # name="" is important to ensure we don't get spurious prefixing
    print("++++++++++++++++++++++++++++++++++++")
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