import tensorflow as tf
from keras.layers import *
from keras_gcn import *
from keras.optimizers import Adam
from keras.models import Sequential,load_model
import os
import keras.backend as K
from keras.models import Model


def export_model(model,
                 export_model_dir,
                 model_version
                 ):
    """
    :param export_model_dir: type string, save dir for exported model
    :param model_version: type int best
    :return:no return
    """
    with tf.get_default_graph().as_default():
        # prediction_signature
        tensor_info_input = tf.saved_model.utils.build_tensor_info(model.input)
        tensor_info_input = tf.saved_model.utils.build_tensor_info(model.output)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': tensor_info_input}, # Tensorflow.TensorInfo
                outputs={'result': tensor_info_input},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        print('step1 => prediction_signature created successfully')
        # set-up a builder
        export_path_base = export_model_dir
        export_path = os.path.join(
            tf.compat.as_bytes(export_path_base),
            tf.compat.as_bytes(str(model_version)))
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            # tags:SERVING,TRAINING,EVAL,GPU,TPU
            sess=K.get_session(),
            tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={'prediction_signature': prediction_signature,},
            )
        print('step2 => Export path(%s) ready to export trained model' % export_path, '\n starting to export model...')
        builder.save(as_text=True)
        print('Done exporting!')

if __name__ == '__main__':
    MapSize = 41
    TimeStep = 3
    X_in = Input(shape=(MapSize,1),name = "StationFeature")
    Map_in = Input(shape=(MapSize,MapSize),name = "Map")
    GCN1 = GraphConv(4,name="GCN1")([X_in,Map_in])
    GCN2 = GraphConv(4,name="GCN2")([GCN1,Map_in])
    Output = Flatten()(GCN2)
    Output = Dense(100,name="ExtractFeature")(Output)

    GCN_Model = Model(inputs = [X_in,Map_in],output = Output,name = "GCN_Part")

    Inputs = []
    Map_Input = []
    GCN_Models = []

    for i in range(TimeStep):
        Inputs.append(Input(shape=(MapSize,1),name = "StationFeature_t-{}".format(TimeStep - i)))
        Map_Input.append(Input(shape=(MapSize,MapSize),name = "Map_t-{}".format(TimeStep - i)))
        GCN_Models.append(GCN_Model([Inputs[i],Map_Input[i]]))

    MergeLayres = concatenate([GCN_Models[i] for i in range(TimeStep)])
    LSTM_Input = Reshape((TimeStep,-1))(MergeLayres)
    LSTM1 = LSTM(64,activation="relu",return_sequences=True,name = "LSTM1")(LSTM_Input)
    LSTM2 = LSTM(64,activation="relu",name="LSTM2")(LSTM1)
    Predict = Dense(1,name="Predict")(LSTM2)

    model = Model(inputs = Inputs + Map_Input,output = Predict,name = "GCN+LSTM")
    model.compile(loss='mse', optimizer=Adam(lr=0.01))
    model.load_weights("{}-minute forecast/models/Station{}.h5".format(15,46))
    model.summary()
    export_model(
        model,
        './export_model',
        1
    )
