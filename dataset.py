import torch 
import torchvision
import torchvision.transforms.functional as F
import os
import shutil
import pandas as pd
import kornia as K
import numpy as np
import math
import pickle

from utils.utils import draw_gaussian, gaussian_radius

VOC_NAMES = ['__background__', "aeroplane", "bicycle", "bird", "boat",
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
            "train", "tvmonitor"]

def read_boxes(label_path):
    '''
    [x1, y1, x2, y2]
    '''
    boxes = []
    with open(label_path) as f:
        for label in f.readlines():
            class_label, x, y, w, h = [
                float(x) if float(x) != int(float(x)) else int(x)
                for x in label.replace("\n", "").split()
            ]
            boxes.append([class_label, x-w/2, y-h/2, x+w/2, y+h/2])
    return boxes

def Hflip_boxes(boxes):
    flipped_boxes = []
    for box in boxes:
        class_label, x1, y1, x2, y2 = box
        x1 = 1 - x1
        x2 = 1 - x2
        x1, x2 = x2, x1
        flipped_boxes.append([class_label, x1, y1, x2, y2])
    return flipped_boxes

class VOCDataset(torch.utils.data.Dataset):

    def __init__(self,
                 root_dir, 
                 csv_file,
                 spatial_resolution,
                 mean = [0.485, 0.456, 0.406], 
                 std = [0.229, 0.224, 0.225],
                 augmentation = False,
                 cache_dir = None,
                 cache_refresh=True,
                 ) -> None:
        super().__init__()
        self.img_dir = os.path.join(root_dir, 'images')
        self.label_dir = os.path.join(root_dir, 'labels')
        csv_file = os.path.join(root_dir, csv_file)
        self.annotations = pd.read_csv(csv_file)

        self.spatial_resolution = spatial_resolution
        self.augmentation = augmentation
        self.mean = mean
        self.std = std

        self.max_objs = 100
        self.down_ratio = 4
        self.num_classes = 20
        self.spatial_detection_h = spatial_resolution[0] // self.down_ratio
        self.spatial_detection_w = spatial_resolution[1] // self.down_ratio
        self.gaussian_iou = 0.7

        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            if cache_refresh == True or os.path.exists(self.cache_dir) == False:
                shutil.rmtree(self.cache_dir, ignore_errors=True)
                os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.annotations)

    def cal_groundtruths(self, boxes):
        hmap_tl = np.zeros((self.num_classes, self.spatial_detection_h, self.spatial_detection_w), dtype=np.float32)
        hmap_br = np.zeros((self.num_classes, self.spatial_detection_h, self.spatial_detection_w), dtype=np.float32)
        regs_tl = np.zeros((self.max_objs, 2), dtype=np.float32)
        regs_br = np.zeros((self.max_objs, 2), dtype=np.float32)

        inds_tl = np.zeros((self.max_objs,), dtype=np.int64)
        inds_br = np.zeros((self.max_objs,), dtype=np.int64)

        num_objs = np.array(min(len(boxes), self.max_objs))
        # ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)
        # ind_masks[:num_objs] = 1
        ind_masks = torch.zeros(self.max_objs, dtype=torch.uint8)
        ind_masks[:num_objs] = 1
        ind_masks = ind_masks.bool()
        
        for i, (label, x1, y1, x2, y2) in enumerate(boxes):
            x1 *= self.spatial_detection_w
            x2 *= self.spatial_detection_w
            y1 *= self.spatial_detection_h
            y2 *= self.spatial_detection_h

            x2 = max(min(x2, self.spatial_detection_w-1), 0)
            x1 = max(min(x1, self.spatial_detection_w-1), 0)
            y2 = max(min(y2, self.spatial_detection_h-1), 0)
            y1 = max(min(y1, self.spatial_detection_h-1), 0)
            
            width, height = math.ceil(x2-x1), math.ceil(y2-y1)
            radius = max(0, int(gaussian_radius((height, width), self.gaussian_iou)))
            assert width > 0 , f'width should be greater than 0, {x2}-{x1}'
            assert height > 0 , f'height should be greater than 0 {y2}-{y1}'
            
            ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
            draw_gaussian(hmap_tl[label], [ix1, iy1], radius)
            draw_gaussian(hmap_br[label], [ix2, iy2], radius)

            regs_tl[i, :] = [x1-ix1, y1-iy1]
            regs_br[i, :] = [x2-ix2, y2-iy2]
            inds_tl[i] = iy1 * self.spatial_detection_w + ix1
            inds_br[i] = iy2 * self.spatial_detection_w + ix2

            assert iy1 < self.spatial_detection_h, f'Incorrect iy1={iy1} should be less than {self.spatial_detection_h}'
            assert ix1 < self.spatial_detection_w, f'Incorrect ix1={ix1} should be less than {self.spatial_detection_w}'
            assert iy2 < self.spatial_detection_h, f'Incorrect iy2={iy2} should be less than {self.spatial_detection_h}'
            assert ix2 < self.spatial_detection_w, f'Incorrect ix2={ix2} should be less than {self.spatial_detection_w}'
            assert inds_br[i] < self.spatial_detection_w * self.spatial_detection_h, f'Incorrect inds_br={inds_br[i]} should be less than {self.spatial_detection_w * self.spatial_detection_h}'
            assert inds_tl[i] < self.spatial_detection_w * self.spatial_detection_h, f'Incorrect inds_tl={inds_tl[i]} should be less than {self.spatial_detection_w * self.spatial_detection_h}'
        return hmap_tl, hmap_br, regs_tl, regs_br, inds_tl, inds_br, ind_masks

    def __getitem__(self, index):
        
        cache_file = os.path.join(self.cache_dir, f'{index}.pkl') if self.cache_dir is not None else ''
        if self.cache_dir is not None and os.path.isfile(cache_file):
            with open(cache_file, 'rb') as f:
                pkl_dict = pickle.load(f)
            image = pkl_dict['image']
            boxes = pkl_dict['boxes']
        else:
            label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
            boxes = read_boxes(label_path) 

            img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
            image = torchvision.io.read_image(img_path)  # CxHxW / torch.uint8

            # Resize
            image = F.resize(image, self.spatial_resolution)

            # Save to cache
            if self.cache_dir is not None:
                with open(cache_file, 'wb') as f:
                    pickle.dump(
                        {
                            "image": image,
                            "boxes": boxes,
                        },
                        f
                    )

        if self.augmentation:
            # Flip
            if np.random.random() > 0.5:
                image = K.geometry.transform.hflip(image)
                boxes = Hflip_boxes(boxes)

        # TODO color jitter (optional)
        # normalize 
        image = (image - image.min()) / (image.max() - image.min())
        image = F.normalize(image, self.mean, self.std)

        # Calculate the ground truth labels
        hmap_tl, hmap_br, regs_tl, regs_br, inds_tl, inds_br, ind_masks = self.cal_groundtruths(boxes)

        return {
            'image': image, 
            'hmap_tl': hmap_tl, 'hmap_br': hmap_br, #(num_classes, h//downratio, w//downratio)
            'regs_tl': regs_tl, 'regs_br': regs_br, #(max_objs, 2)
            'inds_tl': inds_tl, 'inds_br': inds_br, #(max_objs)
            'ind_masks': ind_masks #(max_objs)
        }

def benchmark(N=None):
    import time
    N = len(dataset) if N is None else N
    time_list = []
    for i in range(N):
        start_time = time.time()
        data = dataset.__getitem__(i)
        time_taken = time.time() - start_time
        print(f'{i} has taken %0.4f s' % time_taken)
        time_list.append(time_taken)
    time_array = np.array(time_list)
    print("Mean : %0.4f" % (time_array.mean()))
    print(f"Total Time for {N} samples : %0.4f" % (time_array.sum()))

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from pprint import pprint
    # from utils.utils import plot_image

    dataset = VOCDataset(
                         root_dir="../VOC100examples",
                         csv_file="100examples.csv",
                         cache_dir='cache',
                         spatial_resolution=[512, 512],
                         augmentation=True
                        )
    print("len ", len(dataset))
    benchmark(12) #0.042 without cache
    benchmark(12) #0.008 with cache

    # data = dataset.__getitem__(69)
    # image = data['image'] 
    # # boxes = data['boxes']
    # hmap_tl = data['hmap_tl']
    # hmap_br = data['hmap_br']
    # ind_masks = data['ind_masks']
    # regs_tl = data['regs_tl']
    # regs_br = data['regs_br']

    # for k in data.keys():
    #     print(k, data[k].shape)

    # plot_image(image.permute(1,2,0), boxes.tolist())

    # for i in range(len(dataset)):
    #     print(i)
    #     data = dataset.__getitem__(i)
