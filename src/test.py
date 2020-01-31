# -*- coding:utf-8 -*-
# name: config
# author: bqh
# datetime:2020/1/14 11:14
# =========================

from argparse import ArgumentParser
import matplotlib.pyplot as plt
from PIL import Image
import random
import glob
import cv2
import numpy as np
import scipy.io as sio
from config import *

import tensorflow as tf
from keras import backend as K
# 设置GPU的使用，一定要在所有使用到keras的地方之前加载
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

from model import MSCNN
from vggmodel import DenseLevelNet



def parse_params():
    """
    解析命令行参数
    :return:
    """
    ap = ArgumentParser()
    ap.add_argument('-s', '--show', default='yes', help='if show test result map')
    args_ = ap.parse_args()
    args_ = vars(args_)
    return args_


def get_samples_malldataset(num):
    """
    获取测试图片
    :return:
    """

    def get_annotation():
        """
        读取2000个图片的注解，得到 每个图片的人数 和 每章图片的所有人坐标
        Annotation按照图片命名顺序
        :return:
        """
        mat_annotation = sio.loadmat(os.path.join(MALL_DataPath, 'mall_gt.mat'))
        count_data, position_data = mat_annotation['count'], mat_annotation['frame'][0]
        return count_data, position_data

    counts_true, _ = get_annotation()
    datasize = len(glob.glob(MALL_DataPath + r'\frames\*'))
    # 从验证集选取测试图片
    samples_index = random.sample([i for i in range(int(0.8 * datasize), datasize)], num)
    samples = [glob.glob(MALL_DataPath + r'\frames\*')[i] for i in samples_index]
    images = []
    counts = []
    for i in range(num):
        filename = samples[i]
        img = cv2.resize(cv2.imread(filename), (224, 224)) / 255.
        img = np.expand_dims(img, axis=0)
        images.append(img)
        counts.append(counts_true[samples_index[i]])
    return images, counts


def get_samples_crowdataset(num):
    import random

    mat_annotation = sio.loadmat(os.path.join(MyDataPath, 'crow.mat'))
    counts_true, position_data = mat_annotation['count'], mat_annotation['points']
    filenames = mat_annotation['img']
    samples_index = [i for i in range(len(filenames))]
    random.shuffle(samples_index)
    samples_index = samples_index[0:num]
    images = []
    counts = []
    for index in samples_index:
        filename = os.path.join(MyDataPath, 'img', filenames[index])
        img = imopen(filename)
        img = np.expand_dims(img, axis=0)
        images.append(img)
        counts.append(counts_true[0][index])
    return images, counts


def get_samples_shanghaitech(num):
    """
    获取测试图片
    :return:
    """

    def get_annotation(index):
        """
        读取2000个图片的注解，得到 每个图片的人数 和 每章图片的所有人坐标
        Annotation按照图片命名顺序
        :return:
        """
        mat_data = sio.loadmat(Shanghaitech_DataPath + r'\part_A\test_data\ground_truth\GT_IMG_{}.mat'.format(index))
        position_data, count_data = mat_data['image_info'][0][0][0][0][0], mat_data['image_info'][0][0][0][0][1][0][0]
        return count_data, position_data

    datasize = len(glob.glob(Shanghaitech_DataPath + r'\part_A\test_data\images\*.jpg'))
    # 从验证集选取测试图片
    samples_index = random.sample([i for i in range(datasize)], num)
    samples = [Shanghaitech_DataPath + r'\part_A\test_data\images\IMG_{0}.jpg'.format(i) for i in samples_index]
    images = []
    counts = []
    for i in range(num):
        filename = samples[i]
        img = cv2.resize(cv2.imread(filename), (224, 224)) / 255.
        img = np.expand_dims(img, axis=0)
        images.append(img)
        counts.append(get_annotation(samples_index[i])[0])
    return images, counts


def plot_sample(raw_images, maps, counts, true_counts):
    plt.figure(figsize=(15, 9))
    for i in range(len(maps)):
        plt.subplot(2, 5, i + 1)
        plt.imshow(np.squeeze(raw_images[i], axis=0))
        plt.title('people true num {}'.format(int(true_counts[i])))
        plt.subplot(2, 5, i + 1 + 5)
        plt.imshow(maps[i][0])
        plt.title('people pred num {}'.format(counts[i]))
    plt.show()
    # plt.savefig(os.path.join(RESULT_PATH, 'rst.png'))


def save_result(raw_images, maps, counts, args_, true_counts):
    if not os.path.exists('../results'):
        os.mkdir('../results')
    if args_['show'] == 'yes':
        plot_sample(raw_images, maps, counts, true_counts)


def test(args_):
    model = MSCNN((224, 224, 3))
    model_file = os.path.join(MODEL_PATH, 'mscnn_model_weights.h5')
    if os.path.exists(model_file):
        model.load_weights(model_file, by_name=True)
        samples, true_counts = get_samples_crowdataset(5)
        maps = []
        counts = []
        for sample in samples:
            dmap = model.predict(sample)
            dmap[dmap < 0.01] = 0
            dmap = np.squeeze(dmap, axis=-1)
            counts.append(int(np.sum(dmap)))
            maps.append(dmap)
        # print(counts)
        save_result(samples, maps, counts, args_, true_counts)
    else:
        print("Sorry, cannot find model file in root_path/models/, please download my model or train your model")


def predict():
    import glob
    from tqdm import tqdm

    data_dir = r'E:\code\kesci\crowdcount\dataset\test'
    VGG_Model = r'E:\code\kesci\crowdcount\src\MSCNN-master\models\vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
    Dense_Model = r'E:\code\kesci\crowdcount\src\MSCNN-master\models\dense_model_weights.h5'
    Mscnn_Model = os.path.join(MODEL_PATH, 'mscnn_model_weights.h5')

    images = glob.glob(data_dir + r'\*.jpg')
    res = []

    # 密度等级分类模型
    dense_net = DenseLevelNet(VGG_Model, Dense_Model)
    dense_model = dense_net.model()
    dense_model.load_weights(Dense_Model, by_name=True)

    # 具体人数分类模型
    crow_model = MSCNN((224, 224, 3))
    crow_model.load_weights(Mscnn_Model)

    for img_name in tqdm(images):
        try:
            img = imopen(img_name)
            img = np.expand_dims(img, axis=0)
            dense_prob = dense_model.predict(img)
            dense_level = np.argmax(dense_prob, axis=1)
            dense_level = dense_level[0]
            if dense_level == 0:
                crow_count = 0
            elif dense_level == 2:
                crow_count = 100
            else:
                dmap = crow_model.predict(img)
                dmap[dmap < 0.01] = 0
                dmap = np.squeeze(dmap, axis=-1)
                crow_count = int(np.sum(dmap))
                if crow_count > 100:
                    crow_count = 100
            res.append([os.path.split(img_name)[1], crow_count])
        except Exception as e:
            print(img_name)
            res.append([os.path.split(img_name)[1], -1])

    with open(r'E:\code\kesci\crowdcount\result\dense_res.csv', 'w') as sw:
        for item in res:
            sw.write('{0},{1}\n'.format(item[0], item[1]))


def imopen(im_path):
    im = Image.open(im_path).convert('RGB')
    im = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    img = cv2.resize(im, (224, 224)) / 255.
    return img


if __name__ == '__main__':
    predict()
    # args = parse_params()
    # test(args)
