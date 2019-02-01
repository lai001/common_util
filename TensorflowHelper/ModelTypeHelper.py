import operator
import sys

import tensorflow as tf
import tfcoreml
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile


def inspect_pb_file(model_pb_file_path: str, output_txt_file_path: str):
    if model_pb_file_path is None or model_pb_file_path is '':
        raise ValueError('model_pb_file_path parameter can not be empty')

    if output_txt_file_path is None or output_txt_file_path is '':
        raise ValueError('output_txt_file_path parameter can not be empty')

    graph_def = graph_pb2.GraphDef()
    with open(model_pb_file_path, "rb") as f:
        graph_def.ParseFromString(f.read())

    tf.import_graph_def(graph_def)

    sess = tf.Session()
    OPS = sess.graph.get_operations()

    ops_dict = {}

    sys.stdout = open(output_txt_file_path, 'w')
    for i, op in enumerate(OPS):
        print(
                '---------------------------------------------------------------------------------------------------------------------------------------------')
        print("{}: op name = {}, op type = ( {} ), inputs = {}, outputs = {}".format(i, op.name, op.type, ", ".join(
                [x.name for x in op.inputs]), ", ".join([x.name for x in op.outputs])))
        print('@input shapes:')
        for x in op.inputs:
            print("name = {} : {}".format(x.name, x.get_shape()))
        print('@output shapes:')
        for x in op.outputs:
            print("name = {} : {}".format(x.name, x.get_shape()))
        if op.type in ops_dict:
            ops_dict[op.type] += 1
        else:
            ops_dict[op.type] = 1

    print(
            '---------------------------------------------------------------------------------------------------------------------------------------------')
    sorted_ops_count = sorted(ops_dict.items(), key=operator.itemgetter(1))
    print('OPS counts:')
    for i in sorted_ops_count:
        print("{} : {}".format(i[0], i[1]))


def convert_pb_to_core_model(tf_model_path: str, mlmodel_path: str, output_feature_names: list,
                             input_name_shape_dict: dict = None):
    tfcoreml.convert(tf_model_path=tf_model_path,
                     mlmodel_path=mlmodel_path,
                     output_feature_names=output_feature_names,
                     input_name_shape_dict=input_name_shape_dict)


def convert_pb_to_pbtxt(filename: str, logdir: str, output_filename: str):
    with gfile.FastGFile(filename, 'rb') as f:
        graph_def = tf.GraphDef()

        graph_def.ParseFromString(f.read())

        tf.import_graph_def(graph_def, name='')

        tf.train.write_graph(graph_def, logdir, output_filename, as_text=True)


def generate_pb_file(session: tf.Session, save_path: str = './model/model.pb', logdir: str = './model',
                     output_filename: str = 'model.pbtxt'):
    """
    for exmaple:
        generate_pb_file(session, './model/model.pb', './model', 'model.pbtxt')

    :param session:
    :param save_path:
    :param logdir:
    :param output_filename:
    :return:
    """

    if session is None:
        raise ValueError('session parameter can not ba empty')

    constant_graph = graph_util.convert_variables_to_constants(session, session.graph_def, ['linear_model'])
    with tf.gfile.FastGFile(save_path, mode='wb') as f:
        f.write(constant_graph.SerializeToString())

    convert_pb_to_pbtxt(save_path, logdir, output_filename)


def freeze_graph(input_checkpoint: str, output_graph: str, output_node_names: list, logdir: str = './model',
                 output_filename: str = 'model_freeze.pbtxt'):
    """
    for example:
        freeze_graph('./model/model.cpkt', './model/model_freeze.pb', ["linear_model"],
                        './model', 'model_freeze.pbtxt')

    :param input_checkpoint:
    :param output_graph:
    :param output_node_names:
    :param logdir:
    :param output_filename:
    :return:
    """

    if output_node_names is None or output_node_names is '':
        raise ValueError('output_node_names parameter can not be empty')
    if input_checkpoint is None or output_node_names is '':
        raise ValueError('input_checkpoint parameter can not be empty')
    if output_graph is None or output_node_names is '':
        raise ValueError('output_graph parameter can not be empty')

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
                sess=sess,
                input_graph_def=sess.graph_def,  # 等于:sess.graph_def
                output_node_names=output_node_names)  # 如果有多个输出节点，以逗号隔开

        with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        print("%d ops in the final graph." % len(output_graph_def.node))  # 得到当前图有几个操作节点

    convert_pb_to_pbtxt(output_graph, logdir, output_filename)


def generate_tflite(session: tf.Session, input_tensor_list: list, output_tensor_name_list: list,
                    output_tensor_list: list, output_file_path: str = './converteds_model.tflite'):
    if session is None or input_tensor_list is None or output_tensor_list is None:
        raise ValueError("parameter can not be empty")

    frozen_graphdef = tf.graph_util.convert_variables_to_constants(session, session.graph_def, output_tensor_name_list)
    tflite_model = tf.contrib.lite.toco_convert(frozen_graphdef, input_tensor_list, output_tensor_list)
    open(output_file_path, "wb").write(tflite_model)
