import os
import datetime
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataset import Subset

from modules.loss import cross_entropy2d, multi_scale_cross_entropy2d, bootstrapped_cross_entropy2d
from modules.datasets import Segmentation_dataset
from models.unet import UNet

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_base_path = os.path.join(os.path.dirname(base_path), "datasets", "VOCdevkit", "VOC2012_trainval") # dataset path

original_transform = transforms.Compose([
    torchvision.transforms.Resize((512, 512), interpolation=2),
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
    ])
teacher_transform = transforms.Compose([
    torchvision.transforms.Resize((512, 512), interpolation=2),
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
    ])
val_transform = transforms.Compose([
    torchvision.transforms.Resize((512, 512), interpolation=2),
    transforms.ToTensor(),
    ])


datasets = Segmentation_dataset(dataset_base_path, "JPEGImages", "SegmentationClass", original_transform, teacher_transform, loader=default_image_loader) #  (original_img, anotation_img)

dataloader = torch.utils.data.DataLoader(data_set, batch_size=8, shuffle=True)

# train_size = len(datasets)*9//10
# val_size = len(datasets) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(datasets, [train_size, val_size])

# train_loader = torch.utils.data.DataLoader(train_dataset, )
# val_loader = torch.utils.data.DataLoader(val_dataset, )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = cross_entropy2d()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)


logdir = "logs"
logdir_path = os.path.join(base_path, logdir)
if not os.path.isdir(logdir_path):
    os.mkdir(logdir_path)
dt = datetime.datetime.now()
model_id = len(glob.glob(os.path.join(logdir_path, "{}{}{}*".format(dt.year, dt.month, dt.day))))
log_name = "{}{:02}{:02}_{:02}_{}".format(dt.year, dt.month, dt.day, model_id, model.__class__.__name__)
log_path = os.path.join(logdir_path, log_name)
writer = SummaryWriter(log_dir=log_path)

epochs = 200


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()
        data, target = data.cuda(), target.cuda()
        embedded = model(data)
        loss = criterion(embedded, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("loss/train_loss", loss.item(), (len(train_loader)*(epoch-1)+batch_idx)) # 675*e+i

        if batch_idx % 20 == 0:
            #validation
            model.eval()
            with torch.no_grad():
                val_losses = 0.0
                for idx, (data, target) in enumerate(val_loader):
                    data, target = data.cuda(), target.cuda()
                    embedded = model(data)
                    val_loss = criterion(embedded, target)
                    val_losses += val_loss
            mean_val_loss = val_losses/len(val_loader)
            writer.add_scalar("loss/val_loss", loss.item(), (len(train_loader)*(epoch-1)+batch_idx))
            print('Train Epoch: {:>3} [{:>5}/{:>5} ({:>3.0f}%)]\ttrain_loss: {:>2.4f}\tval_loss: {:>2.4f}'.format(
                epoch,
                batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                loss.item(), val_loss))




def save(epoch):
    checkpoint_path = os.path.join(base_path, "checkpoints")
    save_file = "checkpoint.pth.tar"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(os.path.join(checkpoint_path, log_dir_name)):
        os.makedirs(os.path.join(checkpoint_path, log_dir_name))
    save_path = os.path.join(checkpoint_path, log_dir_name, save_file)
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    for epoch in range(1, epochs+1):
        train(epoch)
        save(epoch)
