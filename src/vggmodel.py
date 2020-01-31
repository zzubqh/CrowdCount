# -*- coding:utf-8 -*-
# name: vggmodel
# author: bqh
# datetime:2020/1/25 10:44
# =========================

import os
import cv2
from PIL import Image
import numpy as np
from argparse import ArgumentParser
from keras.models import Sequential, Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras import layers
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras.applications.vgg16 import VGG16
from keras.callbacks import TensorBoard

import matplotlib.pyplot as plt
from data import DenseDataset

VGG_Model = '../models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
Dense_Model = '../models/dense_model_weights.h5'


class DenseLevelNet():
    def __init__(self, vgg_weigths_path, weights_path):
        self.vgg_weights_path = vgg_weigths_path
        self.weights_path = weights_path

    def model(self):
        """
        使用VGG16做特征提取网络，去掉原VGG网络中的top层，改为自己的分类层
        为方便统一管理，输入大小同mscnn的输入
        :return: 输出one-hot编码的类别，这里共分成3个类，故输出层的大小是3
        """
        base_model = VGG16(include_top=False,
                           weights=self.vgg_weights_path,
                           input_shape=(224, 224, 3))
        # base_model.summary()
        out = base_model.layers[-1].output
        out = layers.Flatten()(out)
        out = layers.Dense(1024, activation='relu')(out)

        out = BatchNormalization()(out)
        out = layers.Dropout(0.5)(out)
        out = layers.Dense(512, activation='relu')(out)
        out = layers.Dropout(0.3)(out)
        out = layers.Dense(128, activation='relu')(out)
        out = layers.Dropout(0.3)(out)
        out = layers.Dense(3, activation='softmax')(out)
        tuneModel = Model(inputs=base_model.input, outputs=out)

        # vgg层不参与训练
        for layer in tuneModel.layers[:19]:
            layer.trainable = False

        return tuneModel

    def get_callbacks(self):
        """
        设置各种回调函数
        :return:
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-7, verbose=True)
        model_checkpoint = ModelCheckpoint(self.weights_path,
                                           monitor='val_loss',
                                           verbose=True,
                                           save_best_only=True,
                                           save_weights_only=True)
        callbacks = [early_stopping, reduce_lr, model_checkpoint, TensorBoard(log_dir='../tensorlog')]
        return callbacks

    def train(self, args_):
        tuneModel = self.model()
        # 使用交叉熵作为损失函数
        tuneModel.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.Adam(lr=3e-4))
        if os.path.exists(self.weights_path):
            tuneModel.load_weights(self.weights_path, by_name=True)
            print('success load weights ')

        batch_size = int(args_['batch'])
        callbacks = self.get_callbacks()
        history = tuneModel.fit_generator(DenseDataset().gen_train(batch_size, 224),
                                          steps_per_epoch=DenseDataset().get_train_num() // batch_size,
                                          validation_data=DenseDataset().gen_valid(batch_size, 224),
                                          validation_steps=DenseDataset().get_valid_num() // batch_size,
                                          epochs=int(args_['epochs']),
                                          callbacks=callbacks)
        if args_['show'] != 'no':
            self.train_show(history)

    def train_show(self, train_history):
        # 在回调函数中设置过TensorBoard后，也可以直接在TensorBoard中查看各种参数变换情况
        acc = train_history.history['acc']
        val_acc = train_history.history['val_acc']
        loss = train_history.history['loss']
        val_loss = train_history.history['val_loss']
        epochs = range(1, 51)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'r', label='Validation acc')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.legend()
        plt.show()

    def predict(self, imgs):
        model = self.model()
        model.load_weights(self.weights_path, by_name=True)
        y_predict = model.predict(imgs, batch_size=len(imgs))
        return y_predict


def parse_command_params():
    """
    解析命令行参数
    :return:
    """
    parser = ArgumentParser()
    parser.add_argument('-e', '--epochs', default=50, help='how many epochs to fit')
    parser.add_argument('-b', '--batch', default=16, help='batch size of train')
    parser.add_argument('-s', '--show', default='no', help='show train history')
    args_ = parser.parse_args()
    args_ = vars(args_)
    return args_


def get_valData():
    import random
    dataset = DenseDataset()
    val_index = [i for i in range(len(dataset.filenames))]
    random.shuffle(val_index)
    val_index = val_index[0:5]
    imgs = []
    labels = []
    for index in val_index:
        img, y_true = dataset.get_img_data(index, 224)
        imgs.append(img)
        labels.append(y_true)
    return imgs, np.array(labels)


def predict():
    import glob
    import random

    img_dir = r'E:\code\kesci\crowdcount\dataset\test'
    filenames = glob.glob(img_dir + r'\*.jpg')
    index_list = [i for i in range(len(filenames))]
    random.shuffle(index_list)
    index_list = index_list[0:10]
    imgs = []
    for index in index_list:
        im = cv2.imread(filenames[index])
        img = cv2.resize(im, (224, 224)) / 255.
        imgs.append(img)

    model_net = DenseLevelNet(VGG_Model, Dense_Model)
    y_predict = model_net.predict(np.array(imgs))
    y_predict = np.argmax(y_predict, axis=1)
    show_res(imgs, y_predict)


def show_res(imgs, labels):
    label_val = [0, 60, 100]
    plt.figure(figsize=(15, 9))
    for i in range(len(labels)):
        img = imgs[i][:, :, ::-1]
        plt.subplot(2, 5, i + 1)
        plt.imshow(img)
        plt.title('people true num {}'.format(label_val[labels[i]]))
    plt.show()


def imopen(im_path):
    im = Image.open(im_path).convert('RGB')
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    img = cv2.resize(im, (224, 224)) / 255.
    return img


def test():
    import tqdm
    label_path = r'E:\code\kesci\crowdcount\result\dense_level.csv'
    img_dir = r'E:\code\kesci\crowdcount\dataset\test'
    img_name = []
    dense_gt = []
    with open(label_path, 'r') as rf:
        for item in rf:
            val = item.strip().split(',')
            img_name.append(val[0])
            dense_gt.append(int(val[1]))
    y_prob = []
    dense_net = DenseLevelNet(VGG_Model, Dense_Model)
    dense_model = dense_net.model()
    dense_model.load_weights(Dense_Model, by_name=True)
    for name in tqdm.tqdm(img_name):
        img = imopen(os.path.join(img_dir, name))
        img = np.expand_dims(img, axis=0)
        dense_prob = dense_model.predict(img)
        dense_level = np.argmax(dense_prob, axis=1)
        dense_level = dense_level[0]
        y_prob.append(dense_level)
    dense_gt = np.array(dense_gt)
    y_prob = np.array(y_prob)
    acc = np.sum(dense_gt == y_prob)/len(dense_gt)
    print(acc)


if __name__ == '__main__':
    test()
    # args = parse_command_params()
    # net = DenseLevelNet(VGG_Model, Dense_Model)
    # net.train(args)
