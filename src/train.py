# -*- coding:utf-8 -*-
# name: config
# author: bqh
# datetime:2020/1/14 11:14
# =========================
from argparse import ArgumentParser
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import losses
from keras.layers import AveragePooling2D
import numpy as np
import os
import warnings
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.callbacks import TensorBoard
from model import MSCNN
from data import MallDataset, ShanghaitechDataset, CrowDataset
import os

ROOT_DIR = r'E:\code\mscnn'
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6  # 每个GPU现存上届控制在60%以内
session = tf.Session(config=config)
KTF.set_session(session)


def parse_command_params():
    """
    解析命令行参数
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', default=50, help='how many epochs to fit')
    parser.add_argument('-v', '--show', default='yes', help='if show training log')
    parser.add_argument('-b', '--batch', default=16, help='batch size of train')
    parser.add_argument('-d', '--dataset', default='shanghaitechdataset', help='which dataset to train')
    parser.add_argument('-p', '--pretrained', default='no', help='load your pretrained model in folder root/models')
    args_ = parser.parse_args()
    args_ = vars(args_)
    return args_


def get_callbacks():
    """
    设置部分回调
    :return:
    """
    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7, verbose=True)
    models_path = os.path.join(ROOT_DIR, 'models')
    if not os.path.exists(models_path):
        os.mkdir(models_path)
    model_checkpoint = ModelCheckpoint(os.path.join(models_path, 'mscnn_model_weights.h5'), monitor='val_loss',
                                       verbose=True, save_best_only=True, save_weights_only=True)
    callbacks = [early_stopping, reduce_lr, model_checkpoint, TensorBoard(log_dir='../tensorlog')]
    return callbacks


def get_avgpoolLoss(y_true, y_pred, k):
    loss = KTF.mean((abs(AveragePooling2D(pool_size=(k, k), strides=(1, 1))(y_true) -
                AveragePooling2D(pool_size=(k, k), strides=(1, 1))(y_pred)))) / k
    return loss


def denseloss(y_true, y_pred, e=1000):
    Le = KTF.mean(KTF.square(y_pred-y_true), axis=-1)
    Lc = get_avgpoolLoss(y_true, y_pred, 1)
    Lc += get_avgpoolLoss(y_true, y_pred, 2)
    Lc += get_avgpoolLoss(y_true, y_pred, 4)
    shp = KTF.get_variable_shape(y_pred)
    Lc = Lc / (shp[1] * shp[2])
    return Le + e * Lc


def train(args_):
    model = MSCNN((224, 224, 3))
    # model.compile(optimizer=Adam(lr=3e-4), loss=denseloss)
    model.compile(optimizer=Adam(lr=3e-4), loss='mse')
    # load pretrained model
    if args_['pretrained'] == 'yes':
        model.load_weights('../models/mscnn_model_weights.h5')
        print("load model from ../models/")

    callbacks = get_callbacks()

    # 流式读取，一个batch读入内存
    batch_size = int(args_['batch'])
    model.fit_generator(CrowDataset().gen_train(batch_size, 224),
                        steps_per_epoch=CrowDataset().get_train_num() // batch_size,
                        validation_data=CrowDataset().gen_valid(batch_size, 224),
                        validation_steps=CrowDataset().get_valid_num() // batch_size,
                        epochs=int(args_['epochs']),
                        callbacks=callbacks)

    # if args_['dataset'] == 'malldataset':
    #     model.fit_generator(MallDataset().gen_train(batch_size, 224),
    #                         steps_per_epoch=MallDataset().get_train_num() // batch_size,
    #                         validation_data=MallDataset().gen_valid(batch_size, 224),
    #                         validation_steps=MallDataset().get_valid_num() // batch_size,
    #                         epochs=int(args_['epochs']),
    #                         callbacks=callbacks)
    # elif args_['dataset'] == 'shanghaitechdataset':
    #     model.fit_generator(ShanghaitechDataset().gen_train(batch_size, 224),
    #                         steps_per_epoch=ShanghaitechDataset().get_train_num() // batch_size,
    #                         validation_data=ShanghaitechDataset().gen_valid(batch_size, 224),
    #                         validation_steps=ShanghaitechDataset().get_valid_num() // batch_size,
    #                         epochs=int(args_['epochs']),
    #                         callbacks=callbacks)
    # else:
    #     print('not support this dataset')


if __name__ == '__main__':
    args = parse_command_params()
    train(args)
