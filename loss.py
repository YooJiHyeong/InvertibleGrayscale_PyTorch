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
    def __init__(self, device, img_shape, threshold, vgg_layer_idx, c_weight, ls_weight):
        super().__init__()

        self.threshold = threshold
        self.vgg = nn.DataParallel(vgg19(pretrained=True).features[:vgg_layer_idx], output_device=device["images"]).to(device["network"])      # if [:26], forward by conv4_4
        self.l1_loss = nn.L1Loss()

        self.c_weight = c_weight
        self.ls_weight = ls_weight

        self.zeros = torch.zeros(img_shape).to(device["images"])

    def lightness(self, gray_img, original_img):

        def _calc_luminance(img):
            r, g, b = torch.unbind(img, dim=1)
            return (.299 * r) + (.587 * g) + (.114 * b)

        luminance = _calc_luminance(original_img)
        # loss = torch.norm(torch.max(torch.abs(gray_img - luminance) - self.threshold, self.zeros), p=1)
        loss = torch.mean(torch.max(torch.abs(gray_img - luminance) - self.threshold, self.zeros))
        return loss

    def contrast(self, gray_img, original_img):
        with torch.no_grad():
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

        g_var = torch.mean((_calc_var(gray_img.repeat(1, 3, 1, 1))))
        o_var = torch.mean((_calc_var(original_img)))
        loss = abs(g_var - o_var)
        # print("var", g_var.item(), o_var.item())
        return loss

    def forward(self, gray_img, original_img):
        l_loss = self.lightness(gray_img, original_img)
        c_loss = self.contrast(gray_img, original_img)
        ls_loss = self.local_structure(gray_img, original_img)
        # print("light :", l_loss.item(), " | contrast :", self.c_weight, c_loss.item(), " | local :", self.ls_weight, ls_loss.item())
        return l_loss + (self.c_weight * c_loss) + (self.ls_weight * ls_loss)


class QuantizationLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gray_img):
        min_tensor = torch.zeros_like(gray_img).fill_(1)   # fill with maximum value (larger than 255)

        for i in range(0, 256):
            min_tensor = torch.min(min_tensor, torch.abs(gray_img - (i / 127.5 - 1)))

        loss = torch.norm(min_tensor, p=1)
        return loss


class TotalLoss(nn.Module):
    def __init__(self, device, img_shape, threshold=70 / 127, vgg_layer_idx=26, c_weight=1e-7, ls_weight=0.5):
        super().__init__()

        self.i_loss = InvertibilityLoss()
        self.g_loss = GrayscaleConformityLoss(device, img_shape, threshold, vgg_layer_idx, c_weight, ls_weight)
        self.q_loss = QuantizationLoss()

    def forward(self, gray_img, original_img, restored_img, loss_stage):
        i_loss = self.i_loss(original_img, restored_img)
        g_loss = self.g_loss(gray_img, original_img)

        if loss_stage == 1:
            g_weight = 1
            q_loss = 0
        elif loss_stage == 2:
            g_weight = 0.5
            q_loss = self.q_loss(gray_img) * 10

        total_loss = i_loss + (g_loss * g_weight) + q_loss
        print("Total %4f | Invert %4f | Gray %4f | Quant %4f" % (total_loss.item(), i_loss.item(), g_loss.item(), float(q_loss)))
        return total_loss
