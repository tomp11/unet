from PIL import Image
import os
import os.path

import torch.utils.data
import torchvision.transforms as transforms
import numpy as np

# w,hの小さい方で正方形にクロップ
class CenterCrop(object):
    def __init__(self):
        pass

    def __call__(self, image):
        # print(image.size)
        size = min(image.size)
        left, upper =  (image.width - size) // 2, (image.height - size) // 2
        img = transforms.functional.crop(image, upper, left, size, size)
        return img

original_transform = transforms.Compose([
CenterCrop(),
transforms.Resize((512, 512), interpolation=2),
transforms.ToTensor(),
])

# index 255(void) to 0(background)
class void2background(object):
    def __init__(self):
        pass

    def __call__(self, image):
        image_array = np.array(image)
        # index = np.where(img_array == 255)
        # image_array[index] = 0
        image_array = np.where(image_array == 255, 0, image_array)
        return image_array




# transforms.ToTensorはpng画像も正規化しちゃうので
class totensor_without_normalize(object):
    def __init__(self):
        pass

    def __call__(self, image_array):
        image_tensor = torch.LongTensor(image_array)
        return image_tensor


class onehot(object):
    def __init__(self):
        self.n_classes = 21

    def __call__(self, image_tensor):
        h, w = image_tensor.size()
        onehot = torch.LongTensor(self.n_classes, h, w).zero_()
        # print(onehot)
        image_tensor = image_tensor.unsqueeze_(0)
        onehot = onehot.scatter_(0, image_tensor, 1)
        return onehot


teacher_transform = transforms.Compose([
CenterCrop(),
transforms.Resize((512, 512), interpolation=2),
void2background(),
totensor_without_normalize(),
onehot(),
])
