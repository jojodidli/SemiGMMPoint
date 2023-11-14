import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS
from ..data_util import crop_pc, voxelize
from ...transforms.point_transform_cpu import PointsToTensor
import glob
from tqdm import tqdm
import logging
import pickle


VALID_CLASS_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39
]

SCANNET_COLOR_MAP = {
    0: (0., 0., 0.),
    1: (174., 199., 232.),
    2: (152., 223., 138.),
    3: (31., 119., 180.),
    4: (255., 187., 120.),
    5: (188., 189., 34.),
    6: (140., 86., 75.),
    7: (255., 152., 150.),
    8: (214., 39., 40.),
    9: (197., 176., 213.),
    10: (148., 103., 189.),
    11: (196., 156., 148.),
    12: (23., 190., 207.),
    14: (247., 182., 210.),
    15: (66., 188., 102.),
    16: (219., 219., 141.),
    17: (140., 57., 197.),
    18: (202., 185., 52.),
    19: (51., 176., 203.),
    20: (200., 54., 131.),
    21: (92., 193., 61.),
    22: (78., 71., 183.),
    23: (172., 114., 82.),
    24: (255., 127., 14.),
    25: (91., 163., 138.),
    26: (153., 98., 156.),
    27: (140., 153., 101.),
    28: (158., 218., 229.),
    29: (100., 125., 154.),
    30: (178., 127., 135.),
    32: (146., 111., 194.),
    33: (44., 160., 44.),
    34: (112., 128., 144.),
    35: (96., 207., 209.),
    36: (227., 119., 194.),
    37: (213., 92., 176.),
    38: (94., 106., 211.),
    39: (82., 84., 163.),
    40: (100., 85., 144.),
}

SCANNET_COLOR_MAP2 = {
    0: [0., 0., 0.],
    1: [174., 199., 232.],
    2: [152., 223., 138.],
    3: [31., 119., 180.],
    4: [255., 187., 120.],
    5: [188., 189., 34.],
    6: [140., 86., 75.],
    7: [255., 152., 150.],
    8: [214., 39., 40.],
    9: [197., 176., 213.],
    10: [148., 103., 189.],
    11: [196., 156., 148.],
    12: [23., 190., 207.],
    14: [247., 182., 210.],
    15: [66., 188., 102.],
    16: [219., 219., 141.],
    17: [140., 57., 197.],
    18: [202., 185., 52.],
    19: [51., 176., 203.],
    20: [200., 54., 131.],
    21: [92., 193., 61.],
    22: [78., 71., 183.],
    23: [172., 114., 82.],
    24: [255., 127., 14.],
    25: [91., 163., 138.],
    26: [153., 98., 156.],
    27: [140., 153., 101.],
    28: [158., 218., 229.],
    29: [100., 125., 154.],
    30: [178., 127., 135.],
    32: [146., 111., 194.],
    33: [44., 160., 44.],
    34: [112., 128., 144.],
    35: [96., 207., 209.],
    36: [227., 119., 194.],
    37: [213., 92., 176.],
    38: [94., 106., 211.],
    39: [82., 84., 163.],
    40: [100., 85., 144.],
}



@DATASETS.register_module()
class ScanNet(Dataset):
    num_classes = 20
    classes = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
               'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
    gravity_dim = 2
    cmap = [*SCANNET_COLOR_MAP2.values()]
    color_mean = [0.46259782, 0.46253258, 0.46253258]
    color_std =  [0.693565  , 0.6852543 , 0.68061745]
    """ScanNet dataset, loading the subsampled entire room as input without block/sphere subsampling.
    number of points per room in average, median, and std: (145841.0, 158783.87179487178, 84200.84445829492)
    """  
    def __init__(self,
                 data_root='data/ScanNet',
                 split='train',
                 voxel_size=0.04,
                 voxel_max=None,
                 transform=None,
                 loop=1, presample=False, variable=False,
                 n_shifted=1,
                 data_rate = "null"
                 ):
        super().__init__()
        data_root = ""
        self.split = split
        self.voxel_size = voxel_size
        self.voxel_max = voxel_max
        self.transform = transform
        self.presample = presample
        self.variable = variable
        self.loop = loop
        self.n_shifted = n_shifted
        self.buffer = None
        self.pipe_transform = PointsToTensor() 
        self.data_rate = data_rate

        if split == "train" :
            if(data_rate=="null"):
                self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
            elif (data_rate[0]=='r'):
                self.data_list = glob.glob(os.path.join(data_root, split+"_"+self.data_rate, "*.pth"))
            else:
                logging.info("error")
                exit(0)
        elif split == 'val':
            self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
        elif split == 'trainval':
            self.data_list = glob.glob(os.path.join(
                data_root, "train", "*.pth")) + glob.glob(os.path.join(data_root, "val", "*.pth"))
        elif split == 'test':
            self.data_list = glob.glob(os.path.join(data_root, split, "*.pth"))
        else:
            raise ValueError("no such split: {}".format(split))

        logging.info("Totally {} samples in {} set.".format(
            len(self.data_list), split))

        processed_root = os.path.join(data_root, 'processed')
        filename = os.path.join(
            processed_root, f'scannet_{split}_{voxel_size:.3f}.pkl')
        if presample and not os.path.exists(filename):
            np.random.seed(0)
            self.data = []
            for item in tqdm(self.data_list, desc=f'Loading ScanNet {split} split'):
                data = torch.load(item)
                coord, feat, label = data
                coord, feat, label = crop_pc(
                    coord, feat, label, self.split, self.voxel_size, self.voxel_max, variable=self.variable)
                cdata = np.hstack(
                    (coord, feat, np.expand_dims(label, -1))).astype(np.float32)
                self.data.append(cdata)
            npoints = np.array([len(data) for data in self.data])
            logging.info('split: %s, median npoints %.1f, avg num points %.1f, std %.1f' % (
                self.split, np.median(npoints), np.average(npoints), np.std(npoints)))
            os.makedirs(processed_root, exist_ok=True)
            with open(filename, 'wb') as f:
                pickle.dump(self.data, f)
                print(f"{filename} saved successfully")
        elif presample:
            with open(filename, 'rb') as f:
                self.data = pickle.load(f)
                print(f"{filename} load successfully")

    def __getitem__(self, idx):
        data_idx = idx % len(self.data_list)
        if self.presample:
            data_item = self.data[data_idx]
            if(self.buffer is not None):
                data_item = np.concatenate((data_item,self.buffer),axis=0)
                self.buffer=None
            if(data_item.shape[0]>220000):
                self.buffer = data_item[220000:]
                data_item = data_item[:220000]
            coord, feat, label = np.split(data_item, [3, 6], axis=1)
        else:
            data_path = self.data_list[data_idx]
            data = torch.load(data_path)
            coord, feat, label = data[0:3]

        feat = (feat + 1) * 127.5
        label = label.astype(np.long).squeeze()
        data = {'pos': coord.astype(np.float32), 'x': feat.astype(np.float32), 'y': label}


        
        if not self.presample: 
            data['pos'], data['x'], data['y'] = crop_pc(
                data['pos'], data['x'], data['y'], self.split, self.voxel_size, self.voxel_max,
                downsample=not self.presample, variable=self.variable)
        if self.transform is not None:
            data = self.transform(data) 
        data = self.pipe_transform(data)
         
        if 'heights' not in data.keys():
            data['heights'] =  data['pos'][:, self.gravity_dim:self.gravity_dim+1] - data['pos'][:, self.gravity_dim:self.gravity_dim+1].min()
        return data

    def __len__(self):
        return len(self.data_list) * self.loop