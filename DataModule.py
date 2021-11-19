from pytorch_lightning.core import datamodule
import torch
import pytorch_lightning as pl
import os
import shutil

from dataset import VOCDataset

class DataModule(pl.LightningDataModule):
    def __init__(self,
                 root_dir,
                 csv_file,
                 batch_size, 
                 spatial_resolution,
                 mean = [0.485, 0.456, 0.406], 
                 std = [0.229, 0.224, 0.225],
                 num_workers=None, 
                 pin_memory=False,
                 shuffle=True):
        super().__init__()
        self.root_dir = root_dir
        self.csv_file = csv_file

        self.spatial_resolution = spatial_resolution
        
        self.mean = mean 
        self.std = std 

        self.batch_size = batch_size 
        self.num_workers = os.cpu_count() - 1 if num_workers is None else num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle

    # def prepare_data(self) -> None:
    #     shutil.rmtree(self.pkl_dir, ignore_errors=True)
    #     os.mkdir(self.pkl_dir)

    def setup(self, stage = None):
        if stage == "fit" or stage is None:
            self.train_dataset = VOCDataset(
                            root_dir=self.root_dir, #os.path.join(self.root_dir, 'train'), 
                            csv_file=self.csv_file,
                            spatial_resolution=self.spatial_resolution,
                            mean=self.mean,
                            std=self.std,
                            augmentation=True,
                        )

        #     self.val_dataset = ViolenceDetectionDataset(
        #                     root=os.path.join(self.root_dir, 'val'), 
        #                     pkl_dir=os.path.join(self.pkl_dir, 'val'),
        #                     spatial_resolution=self.spatial_resolution,
        #                     spatial_detection_resolution=self.spatial_detection_resolution,
        #                     temporal_dimension=self.temporal_dimension,
        #                     K=self.K, 
        #                     num_classes=self.num_classes, 
        #                     max_objs=self.max_objs,
        #                     mean=self.mean, 
        #                     std=self.std, 
        #                     grayscale=self.grayscale
        #                 )
        # if stage == "test" or stage is None:
        #     self.test_dataset = ViolenceDetectionDataset(
        #                     root=os.path.join(self.root_dir, 'test'), 
        #                     pkl_dir=os.path.join(self.pkl_dir, 'test'),
        #                     spatial_resolution=self.spatial_resolution,
        #                     spatial_detection_resolution=self.spatial_detection_resolution,
        #                     temporal_dimension=self.temporal_dimension,
        #                     K=self.K, 
        #                     num_classes=self.num_classes, 
        #                     max_objs=self.max_objs,
        #                     mean=self.mean, 
        #                     std=self.std, 
        #                     grayscale=self.grayscale
        #                 )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
                    self.train_dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    drop_last=True,
                )

    # def val_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #                 self.val_dataset,
    #                 batch_size=self.batch_size,
    #                 shuffle=False,
    #                 num_workers=self.num_workers,
    #                 pin_memory=self.pin_memory,
    #             )

    # def test_dataloader(self):
    #     return torch.utils.data.DataLoader(
    #                 self.test_dataset,
    #                 batch_size=self.batch_size,
    #                 shuffle=False,
    #                 num_workers=self.num_workers,
    #                 pin_memory=self.pin_memory,
    #             )

if __name__ == '__main__':
    datamodule = DataModule(
        root_dir="../VOC100examples",
        csv_file="100examples.csv",
        spatial_resolution=[512, 512],
        batch_size=1, 
        num_workers=0,
        pin_memory=True,
    )
    datamodule.setup()
    train_dl = datamodule.train_dataloader()

    print('len', len(train_dl))

    for data in train_dl:
        break
    for k in data.keys():
        print(k, data[k].shape)