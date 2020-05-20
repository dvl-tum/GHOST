from torchvision import transforms
import PIL.Image
import torch
import random
import math


def std_per_channel(images):
    images = torch.stack(images, dim = 0)
    return images.view(3, -1).std(dim = 1)


def mean_per_channel(images):
    images = torch.stack(images, dim = 0)
    return images.view(3, -1).mean(dim = 1)


class Identity(): # used for skipping transforms
    def __call__(self, im):
        return im


class ScaleIntensities():
    def __init__(self, in_range, out_range):
        """ Scales intensities. For example [-1, 1] -> [0, 255]."""
        self.in_range = in_range
        self.out_range = out_range

    def __call__(self, tensor):
        tensor = (
            tensor - self.in_range[0]
        ) / (
            self.in_range[1] - self.in_range[0]
        ) * (
            self.out_range[1] - self.out_range[0]
        ) + self.out_range[0]
        return tensor


def make_transform(sz_resize = 256, sz_crop = 224, mean = [128, 117, 104],
        std = [1, 1, 1], rgb_to_bgr = True, is_train = True, 
        intensity_scale = [[0, 1], [0, 255]]):
    return transforms.Compose([
        transforms.Compose([ # train: horizontal flip and random resized crop
            transforms.RandomResizedCrop(sz_crop),
            transforms.RandomHorizontalFlip(),
        ]) if is_train else transforms.Compose([ # test: else center crop
            transforms.Resize(sz_resize),
            transforms.CenterCrop(sz_crop),
        ]),
        transforms.ToTensor(),
        ScaleIntensities(
            *intensity_scale) if intensity_scale is not None else Identity(),
        transforms.Normalize(
            mean=mean,
            std=std,
        ),
        transforms.Lambda(
            lambda x: x[[2, 1, 0], ...]
        ) if rgb_to_bgr else Identity()
    ])


class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img

# transformations for paper Bag of Tricks
def make_transform_bot(sz_crop=256, mean=[0.485, 0.456, 0.406],
                       std=[0.299, 0.224, 0.225], is_train=True):
    normalize_transform = transforms.Normalize(mean=mean, std=std)
    if is_train:
        transform = transforms.Compose([
            transforms.Resize(sz_crop),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(10),
            transforms.RandomCrop(sz_crop),
            transforms.ToTensor(),
            normalize_transform,
            RandomErasing(probability=0.5,
                          mean=(0.4914, 0.4822, 0.4465))
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(sz_crop),
            transforms.ToTensor(),
            normalize_transform
        ])

    return transform
