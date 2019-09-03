import os
import datetime
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Subset

from modules.loss import cross_entropy2d, multi_scale_cross_entropy2d, bootstrapped_cross_entropy2d, BCEDiceLoss
from modules.datasets import Segmentation_dataset
from modules.transforms import original_transform, teacher_transform
from models.unet import UNet

import matplotlib.pyplot as plt
import numpy as np

model_id = "20190829_00_UNet"

checkpoint_path = "./checkpoints/{}/checkpoint.pth.tar".format(model_id)

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataroot = os.path.join(os.path.dirname(base_path), "datasets")
if not os.path.exists(dataroot):
    os.mkdir(dataroot)

datasets = torchvision.datasets.VOCSegmentation(dataroot, year='2012', image_set='train', download=False, transform=original_transform, target_transform=teacher_transform)

train_loader = torch.utils.data.DataLoader(datasets, batch_size=1, shuffle=False)

model = UNet(n_channels=3, n_classes=21).cuda()

model.load_state_dict(torch.load(checkpoint_path))

model.eval()
with torch.no_grad():
    data, target = iter(train_loader).next()
    data = data.cuda()
    output = model(data).data  # [4, 21, 512, 512]




img = Image.fromarray(data[0].detach().cpu().transpose(0, 1).transpose(1, 2).numpy().astype(np.uint8))
y = output[0].detach().cpu()
anno_class_img = Image.fromarray(np.uint8(np.argmax(y.numpy(), axis=0)), mode="P").convert('RGB')
target = Image.fromarray(np.sum(target[0].numpy()*np.arange(0, 21)[:, np.newaxis, np.newaxis], axis=0), mode="P").convert('RGB')


def _get_concat_h(img_lst):
    width, height, h = sum([img.width for img in img_lst]), img_lst[0].height, 0
    dst = Image.new('RGB', (width, height))
    for img in img_lst:
        dst.paste(img, (h, 0))
        h += img.width
    return dst


all_img = _get_concat_h([img, anno_class_img, target])
all_img.save("/ML/unet/inter_file/test.jpg")
