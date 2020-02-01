# -*- coding:utf-8 -*-
# name: config
# author: bqh
# datetime:2020/1/14 11:14
# =========================
from scipy.io import loadmat
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import filters
from config import *

# 生成密度图所使用高斯核大小
Gauss_ksize = 3


# 只包含1-99个人的图片数据集
class CrowDataset(object):
    def __init__(self):
        mat_annotation = loadmat(os.path.join(MyDataPath, 'crow.mat'))
        self.img_dir = r'crowdcount\dataset\train\Mydata\img'
        self.filenames = mat_annotation['img']
        self.counts = mat_annotation['count']
        self.positions = mat_annotation['points']

    def get_train_num(self):
        return int(len(self.filenames) * 0.8)

    def get_valid_num(self):
        return len(self.filenames) - int(len(self.filenames) * 0.8)

    def get_densemap(self, img, img_index, size):
        """
        生成密度图，准备输入神经网络
        :param img
        :param img_index
        :param positions
        :param size 神经网络输入层图片大小
        """
        h, w = img.shape[:-1]
        proportion_h, proportion_w = size / h, size / w  # 输入层需求与当前图片大小对比
        gts = self.positions[0][img_index].copy()
        for i, p in enumerate(self.positions[0][img_index]):
            # 取出每个人的坐标
            now_x, now_y = int(p[0] * proportion_w), int(p[1] * proportion_h)  # 按照输入层要求调整坐标位置
            gts[i][0] = now_x
            gts[i][1] = now_y

        res = np.zeros(shape=[size, size])
        bool_res = (gts[:, 0] < size) & (gts[:, 1] < size)
        for k in range(len(gts)):
            gt = gts[k]
            if bool_res[k] == True:
                res[int(gt[1])][int(gt[0])] = 1

        pts = np.array(list(zip(np.nonzero(res)[1], np.nonzero(res)[0])))
        map_shape = [size, size]
        density = np.zeros(shape=map_shape, dtype=np.float32)
        if len(pts) == 1:
            sigmas = [0]
        else:
            neighbors = NearestNeighbors(n_neighbors=1, algorithm='kd_tree', leaf_size=1200)
            neighbors.fit(pts.copy())
            distances, _ = neighbors.kneighbors()
            sigmas = distances.sum(axis=1) * 0.3

        for i in range(len(pts)):
            pt = pts[i]
            pt2d = np.zeros(shape=map_shape, dtype=np.float32)
            pt2d[pt[1]][pt[0]] = 1
            density += cv2.GaussianBlur(pt2d, (Gauss_ksize, Gauss_ksize), sigmas[i])
            # density += filters.gaussian_filter(pt2d, sigmas[i], mode='constant')
        return density

    def get_img_data(self, index, size):
        """
        读取源文件图片
        :param index 图片下标
        :param size 神经网络输入层尺寸
        :return:
        """
        # _, positions = self.get_annotation()
        img = cv2.imread(os.path.join(self.img_dir, self.filenames[index]))
        density_map = np.expand_dims(self.get_densemap(img, index, size // 4), axis=-1)
        img = cv2.resize(img, (size, size)) / 255.

        return img, density_map

    def gen_train(self, batch_size, size):
        """
        生成数据生成器
        :param batch_size:
        :param size:
        :return:
        """
        index_all = list(range(int(len(self.filenames) * 0.8)))  # 取出所有训练数据下标，默认数据的前80%为训练集

        i, n = 0, len(index_all)
        if batch_size > n:
            raise Exception('Batch size {} is larger than the number of dataset {}!'.format(batch_size, n))

        while True:
            if i + batch_size >= n:
                np.random.shuffle(index_all)
                i = 0
                continue
            batch_x, batch_y = [], []
            for j in range(i, i + batch_size):
                x, y = self.get_img_data(index_all[j], size)
                batch_x.append(x)
                batch_y.append(y)
            i += batch_size
            yield np.array(batch_x), np.array(batch_y)

    def gen_valid(self, batch_size, size):
        index_all = list(range(int(len(self.filenames) * 0.8), len(self.filenames)))

        i, n = 0, len(index_all)
        if batch_size > n:
            raise Exception('Batch size {} is larger than the number of dataset {}!'.format(batch_size, n))

        while True:
            if i + batch_size >= n:
                np.random.shuffle(index_all)
                i = 0
                continue
            batch_x, batch_y = [], []
            for j in range(i, i + batch_size):
                x, y = self.get_img_data(index_all[j], size)
                batch_x.append(x)
                batch_y.append(y)
            i += batch_size

            yield np.array(batch_x), np.array(batch_y)

    def gen_all(self, pic_size):
        x_data = []
        y_data = []
        for i in range(len(self.filenames)):
            image, map_ = self.get_img_data(i, pic_size)
            x_data.append(image)
            y_data.append(map_)
        x_data, y_data = np.array(x_data), np.array(y_data)
        return x_data, y_data

    def show_data(self):
        import random
        index = [i for i in range(100)]
        random.shuffle(index)
        index = index[0:5]
        maps = []
        raw_images = []
        true_counts = []
        counts = []
        for i in index:
            im = cv2.imread(os.path.join(self.img_dir, self.filenames[i]))
            raw_images.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            pixel = self.get_densemap(im, i, 224)
            maps.append(pixel)
            counts.append(int(np.sum(pixel)))
            true_counts.append(self.counts[0][i])

        plt.figure(figsize=(15, 9))
        for i in range(len(maps)):
            plt.subplot(2, 5, i + 1)
            plt.imshow(raw_images[i])
            plt.title('people true num {}'.format(int(true_counts[i])))
            plt.subplot(2, 5, i + 1 + 5)
            plt.imshow(maps[i])
            plt.title('people pred num {}'.format(counts[i]))
        plt.show()


# 密度等级的数据集，分为
# 0： 不包含人头
# 1： 包含1-99个人
# 2： 包含的人数大于100个
class DenseDataset(object):
    def __init__(self):
        mat_annotation = loadmat(r'crowdcount\dataset\denselevel\dense_gt.mat')
        self.img_dir = r'crowdcount\dataset\denselevel\img'
        self.filenames = mat_annotation['img']
        self.dense_level = mat_annotation['dense']

    def get_train_num(self):
        return int(len(self.filenames) * 0.8)

    def get_valid_num(self):
        return len(self.filenames) - int(len(self.filenames) * 0.8)

    def get_img_data(self, index, size):
        img = cv2.imread(os.path.join(self.img_dir, self.filenames[index]))
        density_level = self.dense_level[0][index]
        img = cv2.resize(img, (size, size)) / 255.

        return img, density_level

    def gen_train(self, batch_size, size):
        # 取80%的数据为训练数据
        index_all = list(range(int(len(self.filenames) * 0.8)))

        i, n = 0, len(index_all)
        if batch_size > n:
            raise Exception('Batch size {} is larger than the number of dataset {}!'.format(batch_size, n))

        while True:
            if i + batch_size >= n:
                np.random.shuffle(index_all)
                i = 0
                continue
            batch_x, batch_y = [], []
            for j in range(i, i + batch_size):
                x, y = self.get_img_data(index_all[j], size)
                batch_x.append(x)
                batch_y.append(y)
            i += batch_size
            # return np.array(batch_x), np.array(batch_y)
            yield np.array(batch_x), to_categorical(np.array(batch_y), num_classes=3)

    def gen_valid(self, batch_size, size):
        # 数据集的后20%为验证集数据
        index_all = list(range(int(len(self.filenames) * 0.8), len(self.filenames)))

        i, n = 0, len(index_all)
        if batch_size > n:
            raise Exception('Batch size {} is larger than the number of dataset {}!'.format(batch_size, n))

        while True:
            if i + batch_size >= n:
                np.random.shuffle(index_all)
                i = 0
                continue
            batch_x, batch_y = [], []
            for j in range(i, i + batch_size):
                x, y = self.get_img_data(index_all[j], size)
                batch_x.append(x)
                batch_y.append(y)
            i += batch_size

            yield np.array(batch_x), to_categorical(np.array(batch_y), num_classes=3)

    def gen_all(self, pic_size):
        x_data = []
        y_data = []
        for i in range(len(self.filenames)):
            image, map_ = self.get_img_data(i, pic_size)
            x_data.append(image)
            y_data.append(map_)
        x_data, y_data = np.array(x_data), to_categorical(np.array(y_data), num_classes=3)
        return x_data, y_data

    def show_data(self):
        import random
        index = [i for i in range(100)]
        random.shuffle(index)
        index = index[0:10]
        raw_images = []
        dense_level = []
        for i in index:
            im = cv2.imread(os.path.join(self.img_dir, self.filenames[i]))
            raw_images.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            dense_level.append(self.dense_level[0][i])

        plt.figure(figsize=(15, 9))
        for i in range(len(raw_images)):
            plt.subplot(2, 5, i + 1)
            plt.imshow(raw_images[i])
            plt.title('people dense level {}'.format(int(dense_level[i])))
        plt.show()


if __name__ == '__main__':
    t = CrowDataset()
    t.show_data()
