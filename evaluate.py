import torch
import os
import pytorch_lightning as pl
from DataModule import DataModule
from tqdm.auto import tqdm

from models.hourglass import CornerNetPL
from utils.utils import setup_dir
from utils.model import get_latest_pl_checkpoint, load_state_dict

import numpy as np
import matplotlib.pyplot as plt
from utils.inference import post_process

ckpt_file = os.path.join('checkpoints', 'tiny_hourglass.pt')
device='cuda'

# Create and load model
model = CornerNetPL(
    model_type='tiny_hourglass',
    num_classes=20,
)
model = load_state_dict(model, ckpt_file)
model = model.to(device)
model.eval()
print("Model Loaded")

from dataset import VOC_NAMES as label_dict
datamodule = DataModule(
    root_dir="../VOC100examples",
    # spatial_resolution=[512, 512],
    spatial_resolution=[256, 256],
    batch_size=1, 
    num_workers=0,
    pin_memory=True,
    shuffle=True
)
datamodule.setup()
dl = datamodule.test_dataloader()

output_dir = 'results'
setup_dir(output_dir)
save_count = 0
for batch in tqdm(dl):
    image = batch['image']

    with torch.no_grad():
        image = image.to(model.device)
        output = model(image)[-1]

    output_images = post_process(image, output, label_dict)
    for output_image in output_images:
        plt.imshow(output_image)
        plt.xticks([])
        plt.yticks([])
        save_path = os.path.join(output_dir, f'{save_count}.png')
        save_count += 1
        plt.savefig(save_path)