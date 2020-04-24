import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data

from modules.utils import crop_image2
import cv2

class RegionExtractor():
    def __init__(self, image, samples, opts, Show_image):
        self.image = np.asarray(image)
        self.samples = samples

        self.crop_size = opts['img_size'] 
        self.padding = opts['padding'] # 16
        self.batch_size = opts['batch_test'] # 256

        self.index = np.arange(len(samples))
        self.pointer = 0
        # FIXME:
        self.Show_image = Show_image

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.samples):
            self.pointer = 0
            raise StopIteration
        else:
            # self.pointer 一開始有值 開始進入main loop之後就沒了
            # next pointer : mini batch number.
            next_pointer = min(self.pointer + self.batch_size, len(self.samples))
            index = self.index[self.pointer:next_pointer]
            self.pointer = next_pointer
            regions = self.extract_regions(index, self.Show_image)
            regions = torch.from_numpy(regions)
            return regions
    next = __next__

    # FIXME:
    def extract_regions(self, index, Show_image=False):
        regions = np.zeros((len(index), self.crop_size, self.crop_size, 3), dtype='uint8')
        for i, sample in enumerate(self.samples[index]):
            # sample --> bbox
            # self.image.shape = (288, 352, 3)
            regions[i] = crop_image2(self.image, sample, self.crop_size, self.padding)
            ## FIXME: only for looking the fake images 
            if Show_image == True:
                imageee = [0, 1, 2, 3, 4]
                if i in imageee:
                    print(i)
                    cv2.imshow("patch",regions[i])
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

            # regions[i].shape = (107, 107, 3)
        # Before regions.shape (256, 107, 107, 3)
        regions = regions.transpose(0, 3, 1, 2)
        # After regions.shape (256, 3, 107, 107)
        regions = regions.astype('float32') - 128.
        return regions
