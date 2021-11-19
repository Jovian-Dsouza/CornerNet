import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

# from inference.decode import moc_decode, post_process
from typing import *
from torchvision.utils import make_grid

def make_grid_heatmaps(input_hm, output_hm, nrow=2):
    b, c, h, w = input_hm.shape
    batch_heatmaps = []

    # batch_no = 0
    for batch_no in range(b):
        pred_hm = output_hm[batch_no][0] #.detach()
        org_hm =  input_hm[batch_no][0]
        joined = torch.zeros(1, h, w*2)
        joined[0,..., 0:w] = org_hm
        joined[0,..., w:] = pred_hm
        # Draw boundaries
        joined[0, ..., w] = 1 
        joined[0, ..., 0] = 1 
        joined[0, ..., 2*w-1] = 1
        joined[0, 0 ,:] = 1 
        joined[0, h-1 ,:] = 1 
        batch_heatmaps.append(joined)

    return make_grid(batch_heatmaps, nrow=nrow) 

class VisualizationCallback(Callback):
    def __init__(self):
        super().__init__()
        self._first_val_batch_uninit = True

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if self._first_val_batch_uninit:
            self.sample_val_batch = batch
            self._first_val_batch_uninit = False
    
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0:
            with torch.no_grad():
                sample_outputs = pl_module.model(self.sample_val_batch['input'])[0]
                loss, loss_stats = pl_module.loss(sample_outputs, self.sample_val_batch)
                # sample_outputs['hm'] = sample_outputs['hm'].sigmoid()
            output_heatmap = make_grid_heatmaps(self.sample_val_batch['hm'], sample_outputs['hm'])
            pl_module.logger.experiment.add_image('val_heatmaps', output_heatmap, pl_module.current_epoch)

    # def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #     # if phase == 'val' and batch_idx == 0:
    #     #     output_heatmap = make_grid_heatmaps(batch['hm'], output['hm'])
    #     #     self.logger.experiment.add_image('val_heatmaps', output_heatmap, batch_idx)
    #     # print("\n Callback\n")
    #     # print('epoch', pl_module.current_epoch)
    #     # print('batch_idx', batch_idx)
    #     # print('dataloader_idx', dataloader_idx)
    #     # print('outputs', outputs['hm'].shape)
    #     # print('batch', batch['hm'].shape)
    #     # print('val outputs: {}'.format(outputs))

    #     if batch_idx == 0:
    #         output_heatmap = make_grid_heatmaps(batch['hm'], outputs['hm'])
    #         pl_module.logger.experiment.add_image('val_heatmaps', output_heatmap, pl_module.current_epoch)