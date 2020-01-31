# -*- coding:utf-8 -*-
# name: tools
# author: bqh
# datetime:2020/1/20 17:19
# =========================
import numpy as np
# import shutil
import random
import tqdm
import os
import glob
import json
from scipy.io import savemat

json_dir = r'E:\code\kesci\crowdcount\dataset\train\Mydata\temp\json'
dense_label = r'E:\code\kesci\crowdcount\result\dense_level.csv'
save_dir = r'E:\code\kesci\crowdcount\dataset\denselevel'


class DataTools(object):
    def json_parese(self, json_file):
        pointes = []
        with open(json_file, 'r') as rf:
            obj = json.load(rf)
            shps = obj['shapes']
            img_name = obj['imagePath']
            for item in shps:
                pointes.append([int(item['points'][0][0]), int(item['points'][0][1])])
        return img_name, pointes

    # 创建密度等级，粗分为0,1-99,100以上三个标签
    def create_denselevelLabel(self):
        res_dic = {}
        img_list = []
        dense_list = []

        with open(dense_label, 'r') as rf:
            for item in rf:
                val = item.strip().split(',')
                name = val[0]
                dense_level = int(val[1])
                img_list.append(name)
                dense_list.append(dense_level)

        res_dic['img'] = np.array(img_list)
        res_dic['dense'] = np.array(dense_list)
        savemat(os.path.join(r'E:\code\kesci\crowdcount\dataset\denselevel', 'dense_gt.mat'), res_dic)

    # 创建只包含1 - 99人头数的图片标签
    def create_crowLabel(self):
        res_dic = {}
        img_list = []
        count_list = []
        dense_list = []
        potints_list = []

        json_files = glob.glob(json_dir + r'\*.json')
        for json_file in tqdm.tqdm(json_files):
            img_name, pnts = self.json_parese(json_file)
            crow_cnt = len(pnts)
            img_list.append(img_name)
            count_list.append(crow_cnt)
            dense_list.append(1)
            potints_list.append(pnts)

        # 打乱顺序
        index_list = [i for i in range(len(img_list))]
        random.shuffle(index_list)

        img_temp = []
        count_temp = []
        dense_temp = []
        point_temp = []
        for index in index_list:
            img_temp.append(img_list[index])
            count_temp.append(count_list[index])
            dense_temp.append(dense_list[index])
            point_temp.append(potints_list[index])

        res_dic['img'] = np.array(img_list)
        res_dic['count'] = np.array(count_temp)
        res_dic['dense'] = np.array(dense_list)
        res_dic['points'] = np.array(point_temp)
        savemat(os.path.join(save_dir, 'crow_gt.mat'), res_dic)


DataTools().create_denselevelLabel()
