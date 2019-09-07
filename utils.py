import torch
import torchvision
from tensorboardX import SummaryWriter


class TensorboardLogger(SummaryWriter):
    def __init__(self, log_path):
        super().__init__(log_path)
        self.log_path = log_path

    def log_scalar(self, tag, scalar, global_step):
        # single scalar
        if isinstance(scalar, (int, float)):
            self.add_scalar(tag, scalar, global_step)
        # scalar group
        elif isinstance(scalar, dict):
            self.add_scalars(tag, scalar, global_step)

    def log_image(self, original_img, gray_img, restored_img, epoch, plot_name="Plot"):
        img = torch.cat([original_img[0], gray_img[0].repeat(3, 1, 1), restored_img[0]], 2)
        img = torchvision.utils.make_grid(img, nrow=1)
        self.add_image("orig, gray, restored", img, epoch)
