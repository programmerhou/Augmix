from __future__ import absolute_import

from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch
import augmentations


class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''
    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img
        #print(img.shape)
        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                #print(img.shape)

                return img
        return img
    
    
class AugmentAndMix(object):

    def __init__(self, severity=3, width=3, depth=-1, alpha=1.):
        self.severity = severity
        self.width = width
        self.depth = depth
        self.alpha = alpha

    def apply_op(self, image, op, severity):
        image = np.clip(image[0] * 255., 0, 255).astype(np.uint8)
        #image = image.transpose(2,1,0)
        pil_img = Image.fromarray(image)  # Convert to PIL.Image
        pil_img = op(pil_img, severity)
        return np.asarray(pil_img) / 255.

    def normalize(self, image):

        #image = image.transpose(2, 0, 1)  # Switch to channel-first
        MEAN = [0.4914]
        STD = [0.2023]
        mean, std = np.array(MEAN), np.array(STD)
        image = (image - mean[:, None, None]) / std[:, None, None]
        #print(image.shape)
        #print(image)        
        return image

    def __call__(self, image):
        image = image.numpy()
        ws = np.float32(np.random.dirichlet([self.alpha] * self.width))
        m = np.float32(np.random.beta(self.alpha, self.alpha))
        mix = np.zeros_like(image)
        for i in range(self.width):
            image_aug = image.copy()
            d = self.depth if self.depth > 0 else np.random.randint(1, 4)
            for _ in range(d):
                op = np.random.choice(augmentations.augmentations)
                #print(image_aug.shape)
                #image_aug = self.apply_op(image_aug, op, self.severity)
                #print(image_aug.shape)

                #image_aug = self.normalize(image_aug)
                
                #image_aug = np.expand_dims(image_aug, axis=0)
                #image_aug = image_aug[:,:28,:28]
            mix += ws[i] * image_aug
        
        #mix = mix[:,:28,:28]
        mixed = (1 - m) * image + m * mix
        mixed = torch.from_numpy(mixed)
        #print(mixed.shape)
        #print(mixed)
        return mixed
