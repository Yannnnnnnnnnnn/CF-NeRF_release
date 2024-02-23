from utils import find_files

import os
import cv2
from tqdm import tqdm

import torch
from torch.utils.data.dataset import Dataset

class NeRFMMDataset(Dataset):

    def __init__(self, data_dir, pyramid_level=1):
        super().__init__()

        #TODO:
        # remove find_files by os.listdir
        img_paths = find_files(data_dir)
        self.data_dir = data_dir
        self.pyramid_level = pyramid_level

        self.img_paths = img_paths
        self.imgs = []
        assert len(self.img_paths) > 0, "no object in the data directory: [{}]".format(data_dir)

        self.frames_id = []
        for i in range(len(self.img_paths)):
            self.frames_id.append(i)
        print('self.frames_id',self.frames_id)

        #------------
        # load all imgs into memory
        #------------
        for path in tqdm(self.img_paths, '=> Loading data...'):

            # load image
            img = cv2.imread(path,-1)
        
            # remove last channel
            if img.shape[2] == 4:
                img = img[:,:,0:3]
                # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            if img.shape[2] == 3:
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            else:
                print('please input correct image')

            # assume all image with the same shape
            self.H0, self.W0, _ = img.shape

            for j in range(pyramid_level):
                img = cv2.pyrDown(img)

            img = torch.from_numpy(img).to('cuda').float()
            img = img/255.0
            self.imgs.append(img)
        
        # assume all image with the same shape
        self.H =  self.imgs[0].shape[0]
        self.W =  self.imgs[0].shape[1]
        self.sx = self.W0*1.0/self.W
        self.sy = self.H0*1.0/self.H
        print('scale:',self.sx,self.sy)

        # process to torch
        for i in range(len(self.imgs)):
            img = self.imgs[i]
            self.imgs[i] = img.reshape([-1, 3]).float()

        print("=> dataset: size [{} x {}] for {} images".format(self.H, self.W, len(self.imgs)))
 
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        return torch.tensor([index]).long(), self.imgs[index]
