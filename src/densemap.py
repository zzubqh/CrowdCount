# -*- coding:utf-8 -*-
# name: densemap
# author: bqh
# datetime:2020/1/29 12:21
# =========================
import numpy as np
import os
import matplotlib.image as mpimg
import scipy.io as sio
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import math
from config import *


def gaussian_filter_density(gt):
    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    neighbors = NearestNeighbors(n_neighbors=4, algorithm='kd_tree', leaf_size=1200)
    neighbors.fit(pts.copy())
    distances, _ = neighbors.kneighbors()
    density = np.zeros(gt.shape, dtype=np.float32)
    type(distances)
    sigmas = distances.sum(axis=1) * 0.075
    for i in range(len(pts)):
        pt = pts[i]
        pt2d = np.zeros(shape=gt.shape, dtype=np.float32)
        pt2d[pt[1]][pt[0]] = 1
        # starttime = datetime.datetime.now()
        density += filters.gaussian_filter(pt2d, sigmas[i], mode='constant')
        # endtime = datetime.datetime.now()
        #
        # interval = (endtime - starttime)
        # print(interval)
    return density


def create_density(gts, d_map_h, d_map_w):
    res = np.zeros(shape=[d_map_h, d_map_w])
    bool_res = (gts[:, 0] < d_map_w) & (gts[:, 1] < d_map_h)
    for k in range(len(gts)):
        gt = gts[k]
        if (bool_res[k] == True):
            res[int(gt[1])][int(gt[0])] = 1
    pts = np.array(list(zip(np.nonzero(res)[1], np.nonzero(res)[0])))
    neighbors = NearestNeighbors(n_neighbors=4, algorithm='kd_tree', leaf_size=1200)
    neighbors.fit(pts.copy())
    distances, _ = neighbors.kneighbors()
    map_shape = [d_map_h, d_map_w]
    density = np.zeros(shape=map_shape, dtype=np.float32)
    sigmas = distances.sum(axis=1) * 0.075
    for i in range(len(pts)):
        pt = pts[i]
        pt2d = np.zeros(shape=map_shape, dtype=np.float32)
        pt2d[pt[1]][pt[0]] = 1
        # starttime = datetime.datetime.now()
        density += filters.gaussian_filter(pt2d, sigmas[i], mode='constant')
    return density


def shanghaitech_test():
    train_img = Shanghaitech_DataPath + r'\part_A\train_data\images'
    train_gt = Shanghaitech_DataPath + r'\part_A\train_data\ground_truth'
    out_path = r'E:\code\kesci\crowdcount\dataset\train\ShanghaiTech\output'
    validation_num = 15

    img_names = os.listdir(train_img)
    num = len(img_names)
    num_list = np.arange(1, num + 1)
    # random.shuffle(num_list)
    global_step = 1
    for i in num_list:
        full_img = train_img + r'\IMG_' + str(i) + '.jpg'
        full_gt = train_gt + r'\GT_IMG_' + str(i) + '.mat'
        img = mpimg.imread(full_img)
        data = sio.loadmat(full_gt)
        gts = data['image_info'][0][0][0][0][0]  # shape like (num_count, 2)
        count = 1

        # fig1 = plt.figure('fig1')
        # plt.imshow(img)
        shape = img.shape
        if (len(shape) < 3):
            img = img.reshape([shape[0], shape[1], 1])

        d_map_h = math.floor(math.floor(float(img.shape[0]) / 2.0) / 2.0)
        d_map_w = math.floor(math.floor(float(img.shape[1]) / 2.0) / 2.0)
        # starttime = datetime.datetime.now()
        # den_map = gaussian_filter_density(res)
        if (global_step == 4):
            print(1)
        den_map = create_density(gts / 4, d_map_h, d_map_w)
        # endtime = datetime.datetime.now()
        # interval = (endtime - starttime).seconds
        # print(interval)

        p_h = math.floor(float(img.shape[0]) / 3.0)
        p_w = math.floor(float(img.shape[1]) / 3.0)
        d_map_ph = math.floor(math.floor(p_h / 2.0) / 2.0)
        d_map_pw = math.floor(math.floor(p_w / 2.0) / 2.0)

        if (global_step < validation_num):
            mode = 'val'
        else:
            mode = 'train'
        py = 1
        py2 = 1
        for j in range(1, 4):
            px = 1
            px2 = 1
            for k in range(1, 4):
                print('global' + str(global_step))
                # print('j' + str(j))
                # print('k' +str(k))
                print('----------')
                if (global_step == 4 & j == 3 & k == 4):
                    print('global' + str(global_step))
                final_image = img[py - 1: py + p_h - 1, px - 1: px + p_w - 1, :]
                final_gt = den_map[py2 - 1: py2 + d_map_ph - 1, px2 - 1: px2 + d_map_pw - 1]
                px = px + p_w
                px2 = px2 + d_map_pw
                if final_image.shape[2] < 3:
                    final_image = np.tile(final_image, [1, 1, 3])
                image_final_name = os.path.join(out_path, mode + '_img', 'IMG_' + str(i) + '_' + str(count) + '.jpg')
                gt_final_name = os.path.join(out_path, mode + '_gt', 'GT_IMG_' + str(i) + '_' + str(count))
                Image.fromarray(final_image).convert('RGB').save(image_final_name)
                np.save(gt_final_name, final_gt)
                count = count + 1
            py = py + p_h
            py2 = py2 + d_map_ph
        global_step = global_step + 1
        fig2 = plt.figure('fig2')
        plt.imshow(den_map)
        plt.show()


if __name__ == '__main__':
    shanghaitech_test()
