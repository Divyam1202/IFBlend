from collections import OrderedDict
import cv2 as cv
import math
import numpy as np
import pytorch_ssim
import torch
import torch.nn.functional as F
from metrics import mse, psnr
from PIL import Image
from skimage.io import imsave
from torch.autograd import Variable
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from pytorch_msssim import ssim


def PRIm(x, level):
    b, c, h, w = x.shape
    osz = (h // level, w // level)
    return F.interpolate(x, size=osz, mode="bicubic")


def cv2pil(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return Image.fromarray(img)


def shuffle_down(x, factor):
    b, c, h, w = x.shape
    assert h % factor == 0 and w % factor == 0, "H and W must be a multiple of {}".format(factor)
    n = x.reshape(b, c, h // factor, factor, w // factor, factor)
    n = n.permute(0, 3, 5, 1, 2, 4).reshape(b, c * factor**2, h // factor, w // factor)
    return n


def shuffle_up(x, factor):
    b, c, h, w = x.shape
    assert c % factor**2 == 0, "C must be a multiple of {}".format(factor**2)
    n = x.reshape(b, factor, factor, c // (factor**2), h, w)
    n = n.permute(0, 3, 4, 1, 5, 2).reshape(b, c // (factor**2), factor * h, factor * w)
    return n


def tensor_to_img(img_tensor):
    img_array = np.moveaxis(255 * img_tensor.cpu().detach().numpy(), 0, -1)
    return np.rint(np.clip(img_array, 0, 255)).astype(np.uint8)


def rgb2gray(image):
    rgb_image = 255 * image
    return 0.299 * rgb_image[0, :, :] + 0.587 * rgb_image[1, :, :] + 0.114 * rgb_image[2, :, :]


def compute_maxchann_map(img_flare, img_free):
    b, c, h, w = img_flare.shape
    maps = []
    for i in range(b):
        img_diff = torch.abs(img_flare[i] - img_free[i])
        img_map = torch.max(img_diff, 0).values
        img_map = (img_map - img_map.min()) / (img_map.max() - img_map.min())
        maps.append(img_map.unsqueeze(0).unsqueeze(0))
    return torch.cat(maps, dim=0)


def normalize_weights_map(wmap):
    wmap = wmap.detach()
    b, c, h, w = wmap.shape
    for i in range(b):
        for j in range(c):
            m, M = wmap[i, j].min(), wmap[i, j].max()
            wmap[i, j] = (wmap[i, j] - m) / (M - m)
    return wmap


def save_checkpoint(ckp_path, model, optimizer, scheduler):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, f"{ckp_path}/checkpoint.pt")


def load_checkpoint(ckp_path, model, optimizer, scheduler):
    checkpoint = torch.load(ckp_path, map_location=torch.device('cpu'))

    # Remove 'module.' prefix if present
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return model, optimizer, scheduler


def validate_model(model_net, val_dataloader, save_disk=False, out_dir=None, lpips=None):
    model_net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Tensor = lambda x: x.to(device)
    model_net = model_net.to(device)

    num_samples = len(val_dataloader)
    smse, spsnr, sssim, slpips = 0, 0, 0, 0

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            input_img = Variable(batch[0].type(Tensor))
            gt_img = Variable(batch[1].type(Tensor))

            out_img = model_net(input_img)
            out_symm = 2 * (torch.clamp(out_img, 0, 1) - 0.5)
            gt_symm = 2 * (gt_img - 0.5)

            lpips_d = lpips(out_symm, gt_symm).item() if lpips is not None else 0
            slpips += lpips_d

            sssim += ssim(gt_img.detach(), out_img.detach(), data_range=1, size_average=True)

            inp_img = tensor_to_img(input_img.detach().squeeze(0))
            gt_img = tensor_to_img(gt_img.detach().squeeze(0))
            out_img = tensor_to_img(out_img.detach().squeeze(0))

            if save_disk:
                imsave(f"{out_dir}/{i}_in.png", inp_img)
                imsave(f"{out_dir}/{i}_out.png", out_img)
                imsave(f"{out_dir}/{i}_gt.png", gt_img)

            smse += mse(gt_img, out_img)
            spsnr += sk_psnr(gt_img, out_img)

    model_net.train()
    return {
        "MSE": smse / num_samples,
        "PSNR": spsnr / num_samples,
        "SSIM": sssim / num_samples,
        "LPIPS": slpips / num_samples
    }
