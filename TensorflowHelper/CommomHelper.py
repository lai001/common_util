import os

import tensorflow as tf


def set_log_level(level: str = '1'):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = level


def gpu_growth_session() -> tf.Session:
    return tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))


def _parse_line_py(line):
    """
    定义解析.csv文件的方法

    :param line:
    :return:
    """
    splits = line.split(b"|")
    return splits


def _parse_function(line):
    """
    解析.csv文件

    :param line:
    :return:
    """
    return tf.py_func(_parse_line_py, [line], [tf.string, tf.string, tf.string, tf.string])


def read_csv_file(file: str) -> tf.Tensor:
    if file is None or file is '':
        raise ValueError("路径不能为空")
    dataset = tf.data.TextLineDataset([file])
    dataset = dataset.map(_parse_function)
    iterator = dataset.make_one_shot_iterator()
    text = iterator.get_next()
    return text
    
def view_gpu_and_cpu():
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())