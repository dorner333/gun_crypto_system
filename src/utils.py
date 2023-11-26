import torch
import os
import codecs
import datetime
import logging
import pickle
from pathlib import Path


class prjPaths:
    def __init__(self, exp_name, overwrite = False,  getDataset=True):
        self.SRC_DIR = Path.cwd()
        self.ROOT_MOD_DIR = self.SRC_DIR.parent
        
        self.LIB_DIR = self.ROOT_MOD_DIR / 'lib'
        self.EXP_DIR = self.LIB_DIR / exp_name

        self.CHECKPOINT_DIR = self.EXP_DIR / 'ckpt'
        self.LOGS_DIR = self.EXP_DIR / 'logs'

        pth_exists_else_mk = lambda path: os.mkdir(path) if not os.path.exists(path) else None

        if not overwrite and os.path.exists(self.EXP_DIR):
            print(f'{self.EXP_DIR} exists but flag overwrite == False')
            raise OSError('File exists')

        pth_exists_else_mk(self.LIB_DIR)
        pth_exists_else_mk(self.EXP_DIR)
        pth_exists_else_mk(self.CHECKPOINT_DIR)
        pth_exists_else_mk(self.LOGS_DIR)


def generate_key(size: int = 128, gpu_available: bool = True):
    if gpu_available:
        return torch.randn(1, size).cuda()
    else:
        return torch.randn(1, size)
    
def generate_key_batch(size: int = 784, batchsize: int = 1, gpu_available: bool = True):
    if gpu_available:
        return torch.randn(batchsize, 1, size).cuda()
    else:
        return torch.randn(batchsize, 1, size)


def get_logger(log_dir, run_type):

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    if not os.path.exists(os.path.join(log_dir, run_type)):
        os.mkdir(os.path.join(log_dir, run_type))

    current_Time = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s:%(name)s:%(message)s")
    fileHandler = logging.FileHandler(os.path.join(log_dir, run_type, "%s_%s.log"%(run_type,current_Time)))
    fileHandler.setFormatter(formatter)

    logger.addHandler(fileHandler)

    return logger