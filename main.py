# TODO : CHECK RANGE

import os
import argparse

import torch
import torch.nn as nn
torch.backends.cudnn.benchmark = True

from loader import Loader
from network import Encoder, Decoder
from loss import TotalLoss
from runner import Runner
import utils


def arg_parse():
    desc = "Invertible Grayscale"
    parser = argparse.ArgumentParser(description=desc)

    # System configuration
    parser.add_argument('--gpus', type=str,
                        # default=",".join(map(str, range(cuda.device_count())))
                        help="Select GPUs (Default : Maximum number of available GPUs)")
    parser.add_argument('--cpus', type=int, default="32",
                        help="Select the number of CPUs")

    # Directories
    parser.add_argument('--save_dir', type=str, required=True,
                        help='Directory name to save the model')

    # Training configuration
    parser.add_argument('--epoch', type=int, default=200, help='epochs')

    parser.add_argument('--batch_train', type=int, default=12, help='size of batch for train')
    parser.add_argument('--batch_test',  type=int, default=1, help='size of batch for test (and validation also)')

    # Optimizer
    parser.add_argument('--lr',   type=float, default=0.0002,
                        help="learning rate")
    parser.add_argument('--betas', type=float, default=(0.5, 0.999), nargs="*",
                        help="betas for Adam optimizer")

    return parser.parse_args()


if __name__ == "__main__":
    arg = arg_parse()

    out_dir = "%s/outs"%(os.getcwd())
    arg.save_dir = "%s/%s"%(out_dir, arg.save_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.gpus

    device = {
        "model":  torch.device(0),
        "output": torch.device(0)
    }

    csv_path  = "../VOC2012/"

    train_path = csv_path + "train_v1.csv"
    test_path  = csv_path + "test_v1.csv"

    os.makedirs(arg.save_dir, exist_ok=True)
    tensorboard = utils.TensorboardLogger("%s/tb"%(arg.save_dir))

    E = nn.DataParallel(Encoder(), output_device=device["output"]).to(device["model"])
    D = nn.DataParallel(Decoder(), output_device=device["output"]).to(device["model"])
    loss = TotalLoss(device)
    optim = torch.optim.Adam(list(E.parameters()) + list(D.parameters()), lr=arg.lr, betas=arg.betas)

    train_loader = Loader(train_path, arg.batch_train, num_workers=arg.cpus, shuffle=True, drop_last=True)
    test_loader  = Loader(test_path,  arg.batch_test,  num_workers=arg.cpus, shuffle=True, drop_last=True, cycle=True)

    runner_config = {"epoch": arg.epoch}

    model = Runner(E, D, loss, optim, train_loader, test_loader, runner_config, device, tensorboard)
    model.train()
