from PIL import Image
import os
import os.path

import torch.utils.data
import torchvision.transforms as transforms
import numpy as np



"""
torchvision.datasets.VOCSegmentationがあったので使わなかった
mirroringとかするときに自分で作ってもいいかも
original directoryとteacher directory
originalは.jpg
teacherは.png
"""

def default_image_loader(path):
    return Image.open(path)

class Segmentation_dataset(torch.utils.data.Dataset):

    def __init__(self, dataset_base_path, original_dirname, teacher_dirname, original_transform, teacher_transform, loader=default_image_loader):
        self.dataset_base_path = dataset_base_path
        self.original_dir_path = os.path.join(dataset_base_path, original_dirname)
        self.teacher_dir_path = os.path.join(dataset_base_path, teacher_dirname)
        # globで絶対パス取り出すと絶対パスで返るのでそっちもありかも
        self.teacher_img_list = os.listdir(self.teacher_dir_path)
        self.original_img_list = [i.split(".")[0] + ".jpg" for i in self.teacher_img_list]

        self.original_transform = original_transform
        self.teacher_transform = teacher_transform
        self.loader = loader

    def __getitem__(self, index):
        # def path2img(path):
        #     img = self.loader(path)
        #     return img
        original_img = self.loader(os.path.join(self.dataset_base_path, self.original_dir_path, self.original_img_list[index]))
        size = size = min(original_img.size)
        left, upper =  (original_img.width - size) // 2, (original_img.height - size) // 2
        # 画像を正方形にするresizeするときに歪まないから
        # このなかでやるのは正方形にするだけ。あとは引数に任せる
        original_img = self.original_transform(transforms.functional.crop(original_img, upper, left, size, size))

        teacher_img = self.loader(os.path.join(self.dataset_base_path, self.teacher_dir_path, self.teacher_img_list[index]))
        teacher_img = self.teacher_transform(transforms.functional.crop(teacher_img, upper, left, size, size))
        return original_img, teacher_img

    def __len__(self):
        return len(self.teacher_img_list)

if __name__ =="__main__":
    dataset_base_path = "D:\\ML\\datasets\\VOCdevkit\\VOC2012_trainval"
    original_dirname = "JPEGImages"
    teacher_dirname = "SegmentationClass"
    transform = transforms.Compose([
        # transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
        # transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
        # transforms.torchvision.transforms.functional.crop(img, i, j, h, w),
        # torchvision.transforms.CenterCrop(size)
        ])
    test_dataset = Segmentation_dataset(dataset_base_path, original_dirname, teacher_dirname, transform, transform)

    import numpy
    from scipy.ndimage.interpolation import map_coordinates
    from scipy.ndimage.filters import gaussian_filter

# unetのaugumentationはこれの変形らしい
    def elastic_transform(image, alpha, sigma, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        print(x.shape)
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

        distored_image = map_coordinates(image, indices, order=1, mode='reflect')
        return distored_image.reshape(image.shape)

    np.set_printoptions(threshold=np.inf)
    img1 = test_dataset[0][1]
    # img.show()

    img_array = np.array(img1)
    # color_img_array = np.where(img_array == 1, 1, img_array)
    img2 = Image.fromarray(img_array, mode="P")
    img_array2 = np.array(img2)
    print()
    # if img_array.all() == img_array2.all():
    #     print("同じだよ")
    img2.show()
