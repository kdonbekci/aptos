import os
import sys
import pickle
import tensorflow as tf
import tensorflow.keras.backend as K
from datetime import datetime
from pytz import timezone    

TZ = 'US/Eastern'
print(f'time zone: {TZ}')
tz = timezone(TZ)

SRC_DIR = os.path.dirname(os.path.realpath(__file__))
DATASETS = ['aptos', 'chf']
DATA_DIR = os.path.join(SRC_DIR, 'data')
DUMP_DIR = os.path.join(SRC_DIR, 'dumps')
MODELS_DIR = os.path.join(SRC_DIR, 'models')
MACHINE ='local'
PREPROCS = ['subtract_median']
IMAGE_SIZES = [224]

def get_time(time_format):
    time_now = datetime.now(tz)
    return time_now.strftime(time_format)

def save_pickle(obj, path):
    with open(path, 'wb+') as f:
        pickle.dump(obj, f)
        
def load_pickle(path):
    with open(path, 'rb+') as f:
        obj = pickle.load(f)
    return obj


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def parse_function(proto):
    # define your tfrecord again. Remember that you saved your image as a string.
    keys_to_features = {'img': tf.io.FixedLenFeature([], tf.string, default_value=''),
                        'diagnosis': tf.io.FixedLenFeature([], tf.int64, default_value=0),
                        'id': tf.io.FixedLenFeature([], tf.string, default_value='')}
    
    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    
    return parsed_features

@tf.function
def r2(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred )) 
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


CUSTOM_FUNCTIONS = {'r2': r2}

def fix_customs(d):
    temp = {}
    for key, item in d.items():
        if isinstance(item, list):
            temp[key] = []
            for x in item:
                temp[key].append(CUSTOM_FUNCTIONS.get(x, x))
        else:
            temp[key] = CUSTOM_FUNCTIONS.get(item, item)
    return temp

