import torch
import torch.nn as nn
from torchvision.models import vgg19


class InvertibilityLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.loss = nn.MSELoss()

    def forward(self, original_img, restored_img):
        return self.loss(original_img, restored_img)


class GrayscaleConformityLoss(nn.Module):
    def __init__(self, device, threshold, vgg_layer_idx, c_weight, ls_weight):
        super().__init__()

        self.threshold = threshold
        self.vgg = vgg19(pretrained=True).features[:vgg_layer_idx].to(device["model"])      # if [:26], forward by conv4_4
        self.l1_loss = nn.L1Loss()

        self.c_weight = c_weight
        self.ls_weight = ls_weight

    def lightness(self, gray_img, original_img):

        def _calc_luminance(img):
            r, g, b = torch.unbind(img, dim=1)
            return (.299 * r) + (.587 * g) + (.114 * b)

        luminance = _calc_luminance(original_img)
        loss = torch.norm(torch.max(torch.abs(gray_img - luminance), 0).values, p=1)
        return loss

    def contrast(self, gray_img, original_img):
        vgg_g = self.vgg(gray_img.repeat(1, 3, 1, 1))
        vgg_o = self.vgg(original_img)
        return self.l1_loss(vgg_g, vgg_o)

    def local_structure(self, gray_img, original_img):

        def _calc_var(img):
            h_diff = img[:, :, 1:, :] - img[:, :, :-1, :]
            w_diff = img[:, :, :, 1:] - img[:, :, :, :-1]

            sum_axis = (1, 2, 3)
            var = torch.abs(h_diff).sum(dim=sum_axis) + torch.abs(w_diff).sum(dim=sum_axis)

            return var

        g_var = _calc_var(gray_img.repeat(1, 3, 1, 1))
        o_var = _calc_var(original_img)
        loss = self.l1_loss(g_var, o_var)
        return loss

    def forward(self, gray_img, original_img):
        l_loss = self.lightness(gray_img, original_img)
        c_loss = self.contrast(gray_img, original_img)
        ls_loss = self.local_structure(gray_img, original_img)
        return l_loss + (self.c_weight * c_loss) + (self.ls_weight * ls_loss)


class QuantizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gray_img):
        min_tensor = torch.zeros_like(gray_img).fill_(256)   # fill with maximum value (larger than 255)

        for i in range(0, 256):
            min_tensor = torch.min(min_tensor, torch.abs(gray_img - i))

        loss = torch.norm(min_tensor, p=1)
        return loss


class TotalLoss(nn.Module):
    def __init__(self, device, threshold=70, vgg_layer_idx=26, c_weight=1e-7, ls_weight=0.5):
        super().__init__()

        self.i_loss = InvertibilityLoss()
        self.g_loss = GrayscaleConformityLoss(device, threshold, vgg_layer_idx, c_weight, ls_weight)
        self.q_loss = QuantizationLoss()

    def forward(self, gray_img, original_img, restored_img):
        total_loss = self.i_loss(original_img, restored_img) + self.g_loss(gray_img, original_img) + self.q_loss(gray_img)
        return total_loss
