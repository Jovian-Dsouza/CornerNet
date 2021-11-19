import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import optim
import torchmetrics

from utils.losses import CornerLoss
# from skylarklabs_autotrainer.module import TrainerModule

class ObjectDetectionModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = CornerLoss()

    def configure_optimizers(self):
        if self.model.hparams.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), self.model.hparams.lr, momentum = 0.9)
        
        elif self.model.hparams.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), self.model.hparams.lr)
        
        elif self.model.hparams.optimizer == 'adamax':
            return optim.Adamax(self.model.parameters(), self.model.hparams.lr)
    
    def run_step(self, phase, batch, batch_idx):
        outputs = self.model(batch['image'])
        loss, loss_stats = self.loss(outputs, batch)
       
        self.log_dict(
            {
                f'{phase}_loss' : loss_stats['loss'], 
                f'{phase}_focal_loss' : loss_stats['focal_loss'], 
                f'{phase}_reg_loss' : loss_stats['reg_loss'], 
                f'{phase}_pull_loss' : loss_stats['pull_loss'], 
                f'{phase}_push_loss' : loss_stats['push_loss'], 
            },
            on_epoch = True,
            prog_bar = True,
            logger=True
        )
        return loss.mean()

    def training_step(self, batch, batch_idx):
        return self.run_step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        self.run_step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        self.run_step("test", batch, batch_idx)

if __name__ == '__main__':
    from model import CenterNet

    channels=[256, 256, 384, 384, 384, 512]
    modules=[2, 2, 2, 2, 2, 4]
    model = CenterNet(
        nstack=2, channels=channels, modules=modules, num_classes=20,
        optimizer='adam',
        lr=2.5e-4,
    )
    model = ObjectDetectionModule(model)

    print(model.model.hparams)
