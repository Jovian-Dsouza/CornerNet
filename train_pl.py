import warnings

from pytorch_lightning import callbacks
from pytorch_lightning.core import datamodule
warnings.filterwarnings("ignore")

from pytorch_lightning import profiler
import torch
import pytorch_lightning as pl

from trainer import ObjectDetectionModule
from models.hourglass import CornerNetPL as HourGlassCornerNetPL
from models.resnet import CornerNetPL as ResCornerNetPL
from DataModule import DataModule
import shutil
import os
from pytorch_lightning.callbacks import ModelCheckpoint

def lr_finder(trainer, model, datamodule):
    '''
    trainer = pl.Trainer(auto_lr_find=True)
    '''
    lr_finder = trainer.tuner.lr_find(model, datamodule=datamodule)
    # fig = lr_finder.plot(suggest=True)
    # fig.show()
    model.model.hparams.lr = lr_finder.suggestion()
    print('Best learning rate : ', model.model.hparams.lr)

def main():
    pl.seed_everything(137, workers=True)
    checkpoint_cb = ModelCheckpoint(
        save_last=True,
        monitor='val_loss',
        mode='min',
        filename='{epoch}-{val_loss:.4f}',
    )
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
    model = ResCornerNetPL(
        model_type='resnet_18',
        num_classes=20,
        optimizer='adam',
        lr=5e-4,
    )
    model = ObjectDetectionModule(model)

    trainer = pl.Trainer(
                        # profiler="advanced",
                        # fast_dev_run=False,
                        # auto_lr_find=True, 
                        max_epochs=1, 
                        precision=32,
                        benchmark=True,
                        gpus=-1,
                        # progress_bar_refresh_rate=20, #for colab
                        limit_train_batches=1,
                        limit_val_batches=1,
                        # default_root_dir='..',
                        callbacks=[checkpoint_cb],
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
    