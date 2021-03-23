# tensorflow == 1.13.1
from keras_gcn import *
import os

#转pb
import shutil
import tensorflow as tf
tf.keras.backend.clear_session()
tf.keras.backend.set_learning_phase(0)
model = tf.keras.models.load_model("keras模型导出/GCN-Station46.h5",custom_objects={"GraphConv":GraphConv},compile=False)

if os.path.exists('./model/1'):
    shutil.rmtree('./model/1')
    
export_path = './model/1'

# Fetch the Keras session and save the model
with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'inputs': model.input},
        outputs={t.name:t for t in model.outputs})
#生成之后目录结构
#.
#└── 1
#    ├── saved_model.pb
#    └── variables
#        ├── variables.data-00000-of-00001
#        └── variables.index
