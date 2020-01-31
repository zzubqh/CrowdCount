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


class MallDataset(object):
    def __init__(self):
        self.filenames = sorted(glob.glob(MALL_DataPath + r'\frames\*.jpg'), key=lambda x: int(x[-8:-4]))

    def get_train_num(self):
        return int(len(self.filenames) * 0.8)

    def get_valid_num(self):
        return len(self.filenames) - int(len(self.filenames) * 0.8)

    def get_annotation(self):
        """
        读取2000个图片的注解，得到 每个图片的人数 和 每章图片的所有人坐标
        Annotation按照图片命名顺序
        :return:
        """
        mat_annotation = loadmat(os.path.join(MALL_DataPath, 'mall_gt.mat'))
        count_data, position_data = mat_annotation['count'], mat_annotation['frame'][0]
        return count_data, position_data

    def get_pixels(self, img, img_index, positions, size):
        """
        生成密度图，准备输入神经网络
        :param img
        :param img_index
        :param positions
        :param size 神经网络输入层图片大小
        """
        h, w = img.shape[:-1]
        proportion_h, proportion_w = size / h, size / w  # 输入层需求与当前图片大小对比
        pixels = np.zeros((size, size))

        for p in positions[img_index][0][0][0]:
            # 取出每个人的坐标
            now_x, now_y = int(p[0] * proportion_w), int(p[1] * proportion_h)  # 按照输入层要求调整坐标位置
            if now_x >= size or now_y >= size:
                # 越界则向下取整
                print("Sorry skip the point, its index of all is {}".format(img_index))
            else:
                pixels[now_y, now_x] += 1
        pixels = cv2.GaussianBlur(pixels, (Gauss_ksize, Gauss_ksize), 0)
        return pixels

    def get_img_data(self, index, size):
        """
        读取源文件图片
        :param index 图片下标
        :param size 神经网络输入层尺寸
        :return:
        """
        _, positions = self.get_annotation()
        img = cv2.imread(self.filenames[index])
        density_map = np.expand_dims(self.get_pixels(img, index, positions, size // 4), axis=-1)
        img = cv2.resize(img, (size, size)) / 255.

        return img, density_map

    def gen_train(self, batch_size, size):
        """
        生成数据生成器
        :param batch_size:
        :param size:
        :return:
        """
        _, position = self.get_annotation()
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
        """
        生成数据生成器
        :param batch_size:
        :param size:
        :return:
        """
        _, position = self.get_annotation()
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
        """
        数据生成器
        :param pic_size:
        :return:
        """
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
        index = [i for i in range(50)]
        random.shuffle(index)
        index = index[0:5]
        maps = []
        raw_images = []
        true_counts = []
        count_data, position_data = self.get_annotation()
        for i in index:
            img = cv2.imread(self.filenames[i])
            density_map = self.get_pixels(img, i, position_data, 224 // 4)
            raw_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            maps.append(density_map)
            true_counts.append(count_data[i])

        plt.figure(figsize=(15, 9))
        for i in range(len(maps)):
            plt.subplot(2, 5, i + 1)
            plt.imshow(raw_images[i])
            plt.title('people true num {}'.format(int(true_counts[i])))
            plt.subplot(2, 5, i + 1 + 5)
            plt.imshow(maps[i])
            # plt.title('people pred num {}'.format(counts[i]))
        plt.show()


class ShanghaitechDataset(object):
    def __init__(self, part='A'):
        if part == 'A':
            self.folder = os.path.join(Shanghaitech_DataPath, 'part_A')  # '../data/ShanghaiTech/part_A_final/'
        else:
            self.folder = os.path.join(Shanghaitech_DataPath, 'part_B')  # '../data/ShanghaiTech/part_B_final/'

    def get_annotation(self, folder, index):
        """
        读取图片注解
        :param folder 路径必须是part_A/train_data/这一步
        :param index: 图片索引,1开始
        :return:
        """
        mat_data = loadmat(folder + r'\ground_truth\GT_IMG_{}.mat'.format(index))
        positions, count = mat_data['image_info'][0][0][0][0][0], mat_data['image_info'][0][0][0][0][1][0][0]
        return positions, count

    def get_pixels(self, folder, img, img_index, size):
        """
        生成密度图，准备输入神经网络
        :param folder 当前所在数据目录，该数据集目录较为复杂
        :param img 原始图像
        :param img_index 图片在当前目录下的图片序号，1开始
        :param size 目标图大小，按照模型为img的1/4
        """
        positions, _ = self.get_annotation(folder, img_index)
        h, w = img.shape[0], img.shape[1]
        proportion_h, proportion_w = size / h, size / w  # 输入层需求与当前图片大小对比
        pixels = np.zeros((size, size))

        for p in positions:
            # 取出每个人的坐标
            now_x, now_y = int(p[0] * proportion_w), int(p[1] * proportion_h)  # 按照输入层要求调整坐标位置
            if now_x >= size or now_y >= size:
                # 越界则向下取整
                pass
                # print("Sorry skip the point, image index of all is {}".format(img_index))
            else:
                pixels[now_y, now_x] += 1

        pixels = cv2.GaussianBlur(pixels, (Gauss_ksize, Gauss_ksize), 0)
        return pixels

    def gen_train(self, batch_size, size):
        """
        获取训练数据
        :return:
        """
        folder = os.path.join(self.folder, 'train_data')
        index_all = [i + 1 for i in range(len(glob.glob(folder + r'\images\*.jpg')))]

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
                img = cv2.imread(folder + r'\images\IMG_{}.jpg'.format(index_all[j]))
                density = np.expand_dims(self.get_pixels(folder, img, index_all[j], size // 4), axis=-1)
                img = cv2.resize(img, (size, size)) / 255.
                batch_x.append(img)
                batch_y.append(density)
            i += batch_size
            yield np.array(batch_x), np.array(batch_y)

    def gen_valid(self, batch_size, size):
        """
        获取验证数据
        :return:
        """
        folder = os.path.join(self.folder, 'test_data')
        index_all = [i + 1 for i in range(len(glob.glob(folder + r'\images\*.jpg')))]

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
                img = cv2.imread(folder + r'\images\IMG_{}.jpg'.format(index_all[j]))
                density = np.expand_dims(self.get_pixels(folder, img, index_all[j], size // 4), axis=-1)
                density = density.reshape([density.shape[0], density.shape[1], -1])
                img = cv2.resize(img, (size, size)) / 255.
                batch_x.append(img)
                batch_y.append(density)
            i += batch_size
            yield np.array(batch_x), np.array(batch_y)

    def get_train_num(self):
        return len(glob.glob(self.folder + r'\train_data' + r'\images\*'))

    def get_valid_num(self):
        return len(glob.glob(self.folder + r'\test_data' + r'\images\*'))


# 只包含1-99个人的图片数据集
class CrowDataset(object):
    def __init__(self):
        mat_annotation = loadmat(os.path.join(MyDataPath, 'crow.mat'))
        self.img_dir = r'E:\code\kesci\crowdcount\dataset\train\Mydata\img'
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
            # starttime = datetime.datetime.now()
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
        """
        生成数据生成器
        :param batch_size:
        :param size:
        :return:
        """
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
        """
        数据生成器
        :param pic_size:
        :return:
        """
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
        mat_annotation = loadmat(r'E:\code\kesci\crowdcount\dataset\denselevel\dense_gt.mat')
        self.img_dir = r'E:\code\kesci\crowdcount\dataset\denselevel\img'
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
