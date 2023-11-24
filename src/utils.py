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

        self.CHECKPOINT_DIR = self.EXP_DIR / 'chkpts'
        self.PERSIST_DIR = self.EXP_DIR / 'persist'
        self.LOGS_DIR = self.EXP_DIR / 'logs'

        pth_exists_else_mk = lambda path: os.mkdir(path) if not os.path.exists(path) else None

        if not overwrite and os.path.exists(self.EXP_DIR):
            print(f'{self.EXP_DIR} exists but flag overwrite == False')
            raise OSError('File exists')

        pth_exists_else_mk(self.LIB_DIR)
        pth_exists_else_mk(self.EXP_DIR)
        pth_exists_else_mk(self.CHECKPOINT_DIR)
        pth_exists_else_mk(self.PERSIST_DIR)
        pth_exists_else_mk(self.LOGS_DIR)


def generate_data(gpu_available, batch_size, n):
    if gpu_available:
        return [torch.randint(0, 2, (batch_size, n), dtype=torch.float).cuda()*2-1,
                torch.randint(0, 2, (batch_size, n), dtype=torch.float).cuda()*2-1]
    else:
        return [torch.randint(0, 2, (batch_size, n), dtype=torch.float)*2-1,
                torch.randint(0, 2, (batch_size, n), dtype=torch.float)*2-1]
# end

def UTF_8_to_binary(p_utf_8):

    # utf-8 -> binary
    p_bs = " ".join(format(ord(x), "08b") for x in p_utf_8).split(" ")
    return p_bs
# end

def binary_to_UTF_8(p_bs):

    # binary string -> ord
    p_ords = [int(p_b, 2) for p_b in p_bs]

    # ord -> hex "0x68"[2:] must slice to be valid hex
    p_hexs = [hex(p_ord)[2:] for p_ord in p_ords]

    # hex -> utf-8
    decoded = "".join([codecs.decode(p_hex.strip(), "hex").decode("utf-8") for p_hex in p_hexs])
    return decoded
# end

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
# end

def persist_object(full_path, x):
    with open(full_path, "wb") as file:
        pickle.dump(x, file)
# end

def restore_persist_object(full_path):
    with open(full_path, "rb") as file:
        x = pickle.load(file)
    return x
# end
