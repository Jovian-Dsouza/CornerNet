import torch
import os
from glob import glob

def get_latest_pl_checkpoint(log_dir='lightning_logs'):
    ckpt_list = glob(os.path.join(log_dir, '*', 'checkpoints', '*.ckpt'))
    ckpt_list = sorted(ckpt_list,
                key=lambda x: int(x.split(os.path.sep)[-3].split('_')[-1]), reverse=True)
    ckpt_file = ckpt_list[0]
    print('latest pl checkpoint found :', ckpt_file)
    return ckpt_file

def load_state_dict_with_trainer(model, ckpt_file):
    '''
    Load model which is embedded in trainer class
    '''
    model.load_state_dict(torch.load(ckpt_file)['state_dict'])
    return model 

def load_state_dict(model, ckpt_file):
    '''
    load model without the trainer class
    '''
    state_dict = torch.load(ckpt_file)['state_dict']
    _state_dict = {k.replace('model.', ''):v for k, v in state_dict.items()}
    model.load_state_dict(_state_dict)
    return model 


if __name__ == '__main__':
    get_latest_pl_checkpoint()