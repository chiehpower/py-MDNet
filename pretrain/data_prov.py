import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from modules.sample_generator import SampleGenerator
from modules.utils import crop_image2


class RegionDataset(data.Dataset):
    def __init__(self, img_list, gt, opts):
        """
        只有一開始會用到 
        datasets/VOT/vot2013/cup/img/00000001.jpg
        datasets/VOT/vot2013/bicycle/img/00000001.jpg
        datasets/VOT/vot2013/bolt/img/00000001.jpg
        datasets/VOT/vot2013/car/img/00000001.jpg
        datasets/VOT/vot2013/david/img/00000001.jpg
        datasets/VOT/vot2013/diving/img/00000001.jpg
        datasets/VOT/vot2013/face/img/00000001.jpg
        datasets/VOT/vot2013/gymnastics/img/00000001.jpg
        datasets/VOT/vot2013/hand/img/00000001.jpg
        datasets/VOT/vot2013/iceskater/img/00000001.jpg
        datasets/VOT/vot2013/juice/img/00000001.jpg
        datasets/VOT/vot2013/jump/img/00000001.jpg
        datasets/VOT/vot2013/singer/img/00000001.jpg
        datasets/VOT/vot2013/sunshade/img/00000001.jpg
        datasets/VOT/vot2013/torus/img/00000001.jpg
        datasets/VOT/vot2013/woman/img/00000001.jpg
        """
        self.img_list = np.asarray(img_list)
        self.gt = gt

        # 每一個mini-batch的構成是從某一視訊中隨機採8幀圖片，
        # 在這8幀圖片上隨機採32個正樣本和96個負樣本，
        # 即每個mini-batch由某一個視訊的128個框來構成。
        self.batch_frames = opts['batch_frames'] # 8
        self.batch_pos = opts['batch_pos'] # 32
        self.batch_neg = opts['batch_neg'] # 96

        self.overlap_pos = opts['overlap_pos']
        self.overlap_neg = opts['overlap_neg']

        self.crop_size = opts['img_size']
        self.padding = opts['padding']

        self.flip = opts.get('flip', False)
        self.rotate = opts.get('rotate', 0)
        self.blur = opts.get('blur', 0)

        self.index = np.random.permutation(len(self.img_list))
        self.pointer = 0

        image = Image.open(self.img_list[0]).convert('RGB')
        self.pos_generator = SampleGenerator('uniform', image.size,
                opts['trans_pos'], opts['scale_pos'])
        self.neg_generator = SampleGenerator('uniform', image.size,
                opts['trans_neg'], opts['scale_neg'])

    def __iter__(self):
        return self

    def __next__(self):
        # next pointer : mini batch number.
        next_pointer = min(self.pointer + self.batch_frames, len(self.img_list))
        idx = self.index[self.pointer:next_pointer]
        if len(idx) < self.batch_frames:
            self.index = np.random.permutation(len(self.img_list))
            next_pointer = self.batch_frames - len(idx)
            idx = np.concatenate((idx, self.index[:next_pointer]))
        self.pointer = next_pointer

        pos_regions = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        neg_regions = np.empty((0, 3, self.crop_size, self.crop_size), dtype='float32')
        for i, (img_path, bbox) in enumerate(zip(self.img_list[idx], self.gt[idx])):
            # one img generates 4 (pos) and 12 (neg) samples. 
            image = Image.open(img_path).convert('RGB')
            image = np.asarray(image)

            n_pos = (self.batch_pos - len(pos_regions)) // (self.batch_frames - i) # 4
            n_neg = (self.batch_neg - len(neg_regions)) // (self.batch_frames - i) # 12
            pos_examples = self.pos_generator(bbox, n_pos, overlap_range=self.overlap_pos) # (4,4)
            neg_examples = self.neg_generator(bbox, n_neg, overlap_range=self.overlap_neg) # (12,4)
            pos_regions = np.concatenate((pos_regions, self.extract_regions(image, pos_examples)), axis=0)
            neg_regions = np.concatenate((neg_regions, self.extract_regions(image, neg_examples)), axis=0)

        pos_regions = torch.from_numpy(pos_regions) #  torch.Size([32, 3, 107, 107])
        neg_regions = torch.from_numpy(neg_regions) #  torch.Size([96, 3, 107, 107])
        return pos_regions, neg_regions

    next = __next__

    def extract_regions(self, image, samples):
        regions = np.zeros((len(samples), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(samples):
            regions[i] = crop_image2(image, sample, self.crop_size, self.padding,
                    self.flip, self.rotate, self.blur)
        regions = regions.transpose(0, 3, 1, 2)
        regions = regions.astype('float32') - 128.
        return regions
