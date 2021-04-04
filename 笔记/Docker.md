# 安装docker报错Hardware assisted virtualization and data execution protection must be enabled in the BIOS

https://blog.csdn.net/mythest/article/details/92999646

**解决方法**

其实我这个应该算是 Hyper-V异常导致的，所以要么禁用之后再启用，要么直接运行以下命令,算是重启这个服务：

```
bcdedit /set hypervisorlaunchtype auto
```

之后再重启电脑就ok了，可以愉快地开始docker旅程了。

# TensorFlow Serving入门

## 示例（一）：RESTfull API形式

### 1. 准备TF Serving的Docker环境

目前TF Serving有Docker、APT（二级制安装）和源码编译三种方式，但考虑实际的生产环境项目部署和简单性，推荐使用Docker方式。

```bash
docker pull tensorflow/serving
```

### 2. 下载官方示例代码

示例代码中包含已训练好的模型和与服务端进行通信的客户端（RESTfull API形式不需要专门的客户端）

```bash
mkdir -p /tmp/tfserving
cd /tmp/tfserving
git clone https://github.com/tensorflow/serving
```

### 3. 运行TF Serving

```
docker run -p 8501:8501 --mount type=bind,source=E:\ProgramData\software\serving\tensorflow_serving\servables\tensorflow\testdata\saved_model_half_plus_two_cpu,target=/models/half_plus_two -e MODEL_NAME=half_plus_two -t tensorflow/serving &
```

这里需要注意的是，较早的docker版本没有“--mount”选项，比如Ubuntu16.04默认安装的docker就没有（我的环境是Ubuntu 18.04）。

### 4.客户端验证

```
curl -XPOST http://localhost:8501/v1/models/half_plus_two:predict -d "{\"instances\":[1.0, 2.0, 5.0]}"
```

返回结果，

```
{ "predictions": [2.5, 3.0, 4.5] }
```

# 模型转换成.pb的文件

参考原文：https://blog.csdn.net/mouxiaoqiu/article/details/81220222?utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control&dist_request_id=&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromMachineLearnPai2%7Edefault-1.control

```python
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
```



## 问题：

### 成功解决TypeError: a bytes-like object is required, not 'str'

**解决问题**

TypeError: a bytes-like object is required, not 'str'

**解决思路**

问题出在python3.5和Python2.7在套接字返回值解码上有区别:

python bytes和str两种类型可以通过函数encode()和decode()相互转换，

str→bytes：encode()方法。str通过encode()方法可以转换为bytes。

bytes→str：decode()方法。如果我们从网络或磁盘上读取了字节流，那么读到的数据就是bytes。要把bytes变为str，就需要用decode()方法。 

**解决方法**

将line.strip().split(",")  改为  line.decode().strip().split(",")，大功告成！

**正确写法：**（字符串转16进制）

```python
server_reply = binascii.hexlify(s.recv(1024)).decode()
print(server_reply)
```

## tensorflow中tf.Graph()使用说明

 tf.Graph()表示实例化一个用于tensorflow计算和表示用的数据流图，不负责运行计算。在代码中添加的操作和数据都是画在纸上的画，而图就是呈现这些画的纸。我们可以利用很多线程生成很多张图，但是默认图就只有一张。

  tf中可以定义多个计算图，不同计算图上的张量和运算是相互独立的，不会共享。计算图可以用来隔离张量和计算，同时提供了管理张量和计算的机制。

  1、使用g = tf.Graph()函数创建新的计算图

  2、在with g.as_default():语句下定义属于计算图g的张量和操作

  3、在with tf.Session()中通过参数graph=xxx指定当前会话所运行的计算图

  4、如果没有显示指定张量和操作所属的计算图，则这些张量和操作属于默认计算图

  5、一个图可以在多个sess中运行，一个ses也能运行多个图

  操作示例：

```python
\# 默认计算图上的操作
a = tf.constant([1.0, 2.0])
b = tf.constant([2.0, 3.0])
result = a + b
\# 定义两个计算图
g1 = tf.Graph()
g2 = tf.Graph()
\# 在g1中定义张量和操作
with g1.as_default():
  a = tf.constant([1.0, 1.0])
  b = tf.constant([1.0, 1.0])
  result1 = a + b
\# 在g2中定义张量和操作
with g2.as_default():
  a = tf.constant([2.0, 2.0])
  b = tf.constant([2.0, 2.0])
  result2 = a + b
\# 创建会话
with tf.Session(graph=g1) as sess:
  out = sess.run(result1)
  print(out)

with tf.Session(graph=g2) as sess:
  out = sess.run(result2)
  print(out)

with tf.Session(graph=tf.get_default_graph()) as sess:
  out = sess.run(result)
  print(out)

返回：
[2.0, 2.0]
[4.0, 4.0]
[3.0, 5.0]
```

