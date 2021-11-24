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
    ckpt_dict = torch.load(ckpt_file)
    state_dict = ckpt_dict['state_dict'] if 'state_dict' in ckpt_dict else ckpt_dict
    model.load_state_dict(state_dict)
    return model 

def load_state_dict(model, ckpt_file):
    '''
    load model without the trainer class
    '''
    ckpt_dict = torch.load(ckpt_file)
    state_dict = ckpt_dict['state_dict'] if 'state_dict' in ckpt_dict else ckpt_dict
    _state_dict = {k.replace('model.', ''):v for k, v in state_dict.items()}
    model.load_state_dict(_state_dict)
    return model 

def extract_checkpoint(ckpt_path, save_path):
    '''
    Extracts the state_dict from the checkpoint path
    '''
    state_dict = torch.load(ckpt_path,
                         map_location=lambda storage, loc: storage)['state_dict']
    torch.save({'state_dict' : state_dict}, save_path)
    
if __name__ == '__main__':
    get_latest_pl_checkpoint()