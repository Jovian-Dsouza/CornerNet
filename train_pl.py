import warnings
warnings.filterwarnings("ignore")

from pytorch_lightning import profiler
from tensorboard import data
import torch
import pytorch_lightning as pl

from trainer import ObjectDetectionModule
from model import CornerNetPL
from DataModule import DataModule
import shutil
import os
import pickle

def main():
    datamodule = DataModule(
        root_dir="../VOC100examples",
        # spatial_resolution=[512, 512],
        spatial_resolution=[256, 256],
        batch_size=1, 
        num_workers=0,
        pin_memory=True,
        cache_dir='cache',
        cache_refresh=True,
    )
    model = CornerNetPL(
        model_type='tiny_hourglass',
        num_classes=20,
        optimizer='adam',
        lr=5e-4,
    )
    model = ObjectDetectionModule(model)

    trainer = pl.Trainer(
                        # profiler="advanced",
                        fast_dev_run=False, 
                        max_epochs=1, 
                        precision=32,
                        benchmark=True,
                        gpus=-1,
                        # progress_bar_refresh_rate=20, #for colab
                        limit_train_batches=1,
                        # limit_val_batches=0,
                        # default_root_dir='..',
                        )
    
    trainer.fit(model, datamodule=datamodule)

########################################################
def setup_dir(dir):
    try:
        shutil.rmtree(dir)
    except:
        pass
    os.mkdir(dir)

def launch_tensorboard():
    setup_dir(log_dir)

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', log_dir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    webbrowser.open(url)

if __name__ == '__main__':
    from tensorboard import program
    import webbrowser

    log_dir = 'lightning_logs'
    # launch_tensorboard()
    main()
    