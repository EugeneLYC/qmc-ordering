import torch
import math
import sys
import random
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import collections
import warnings

# from PIL import Image, ImageFilter
from torchvision.transforms import functional as F
import torchvision.datasets as datasets
import qmcpy as qp

# __all__ = ["Compose", "ToTensor", "ToPILImage", "Normalize", "Resize", "CenterCrop", "Pad",
#            "Lambda", "RandomApply", "RandomChoice", "RandomOrder", "RandomCrop", "RandomHorizontalFlip",
#            "RandomVerticalFlip", "RandomResizedCrop", "RandomSizedCrop", "FiveCrop", "TenCrop", "LinearTransformation",
#            "ColorJitter", "RandomRotation", "RandomAffine", "Grayscale", "RandomGrayscale",
#            "RandomPerspective", "RandomErasing"]

__all__ = ["Compose", "ToTensor", "QMCCrop", "QMCHorizontalFlip", "QMCRotation"]


def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))

def unit_interval_to_categorical(x, K):
    c = int(math.floor(K * float(x)))
    if c >= K:
        return K-1
    elif c < 0:
        return 0
    else:
        return c


class QuasiCIFAR10:
    def __init__(self, cifar10: datasets.CIFAR10, transform, args) -> None:
        self.cifar10 = cifar10
        self.transform = transform
        self.state = 0
        self.args = args
        if args.sobol_type == 'independent':
            self.sobolengs = [torch.quasirandom.SobolEngine(dimension=3, scramble=True) for k in range(50000)]
        elif args.sobol_type == 'overlap':
            self.sobolengs = torch.quasirandom.SobolEngine(dimension=3, scramble=True).draw(65536)
        elif args.sobol_type == 'identical':
            num_sample = 256
            self.sobolengs = torch.quasirandom.SobolEngine(dimension=3, scramble=True).draw(num_sample)
        else:
            raise NotImplementedError
        self.sample_visited = {i:0 for i in range(50000)}
    
    def update(self, epoch):
        self.epoch = epoch


    def __getitem__(self, index: int):
        if self.args.sobol_type == 'independent':
            x = self.sobolengs[index].draw()[0]
        elif self.args.sobol_type == 'overlap':
            # x = self.sobolengs[index + self.sample_visited[index]]
            x = self.sobolengs[index + self.epoch]
        elif self.args.sobol_type == 'identical':
            x = self.sobolengs[self.sample_visited[index]]
        else:
            raise NotImplementedError
        # x = self.sobolengs[index].draw()
        s = unit_interval_to_categorical(x[2], 2)
        i = unit_interval_to_categorical(x[0], 9)
        j = unit_interval_to_categorical(x[1], 9)

        (img,target) = self.cifar10.__getitem__(index)
        if s != 0:
            img = F.hflip(img)
        img = F.pad(img, 4)
        img = F.crop(img, i, j, 32, 32)
        if self.transform is not None:
            img = self.transform(img)
        self.state += 1
        if self.args.sobol_type in ['overlap', 'identical']:
            self.sample_visited[index] += 1
        return (img,target)

    def __len__(self) -> int:
        return len(self.cifar10.data)

class SobolNaiveQuasiCIFAR10:
    def __init__(self, cifar10: datasets.CIFAR10, transform, args) -> None:
        self.cifar10 = cifar10
        self.transform = transform
        self.state = 0
        self.args = args
        self.sobolengs = torch.quasirandom.SobolEngine(dimension=1, scramble=True)


    def __getitem__(self, index: int):
        index = int(self.sobolengs.draw()[0][0] * 49999 + 0.5)
        x = random.random()
        y = random.random()
        z = random.random()
        # x = self.sobolengs[index].draw()
        s = unit_interval_to_categorical(x, 2)
        i = unit_interval_to_categorical(y, 9)
        j = unit_interval_to_categorical(z, 9)

        (img,target) = self.cifar10.__getitem__(index)
        if s != 0:
            img = F.hflip(img)
        img = F.pad(img, 4)
        img = F.crop(img, i, j, 32, 32)
        if self.transform is not None:
            img = self.transform(img)
        self.state += 1
        return (img,target)

    def __len__(self) -> int:
        return len(self.cifar10.data)

class SobolCorrQuasiCIFAR10:
    def __init__(self, cifar10: datasets.CIFAR10, transform, args) -> None:
        self.cifar10 = cifar10
        self.transform = transform
        self.state = 0
        self.args = args
        if args.sobol_type == 'independent':
            self.sobolengs = [torch.quasirandom.SobolEngine(dimension=4, scramble=True) for k in range(50000)]
        elif args.sobol_type == 'overlap':
            self.sobolengs = torch.quasirandom.SobolEngine(dimension=4, scramble=True).draw(65536)
        elif args.sobol_type == 'identical':
            num_sample = 256
            self.sobolengs = torch.quasirandom.SobolEngine(dimension=4, scramble=True).draw(num_sample)
        else:
            raise NotImplementedError
        self.sample_visited = {i:0 for i in range(50000)}


    def __getitem__(self, index: int):
        if self.args.sobol_type == 'independent':
            x = self.sobolengs[index].draw()[0]
        elif self.args.sobol_type == 'overlap':
            x = self.sobolengs[index + self.sample_visited[index]]
        elif self.args.sobol_type == 'identical':
            x = self.sobolengs[self.sample_visited[index]]
        else:
            raise NotImplementedError
        # x = self.sobolengs[index].draw()
        index = int(x[3] * 49999 + 0.5)
        s = unit_interval_to_categorical(x[2], 2)
        i = unit_interval_to_categorical(x[0], 9)
        j = unit_interval_to_categorical(x[1], 9)

        (img,target) = self.cifar10.__getitem__(index)
        if s != 0:
            img = F.hflip(img)
        img = F.pad(img, 4)
        img = F.crop(img, i, j, 32, 32)
        if self.transform is not None:
            img = self.transform(img)
        self.state += 1
        if self.args.sobol_type in ['overlap', 'identical']:
            self.sample_visited[index] += 1
        return (img,target)

    def __len__(self) -> int:
        return len(self.cifar10.data)

class QuasiCIFAR100:
    def __init__(self, cifar100, transform, args) -> None:
        self.cifar100 = cifar100
        self.transform = transform
        self.state = 0
        self.args = args
        # num_sample = 65536 if 50000+self.epochs <= 65536 else 65536*2
        if args.sobol_type == 'independent':
            self.sobolengs = [torch.quasirandom.SobolEngine(dimension=4, scramble=True) for k in range(50000)]
        elif args.sobol_type == 'overlap':
            self.sobolengs = torch.quasirandom.SobolEngine(dimension=4, scramble=True).draw(65536)
        elif args.sobol_type == 'identical':
            num_sample = 256
            self.sobolengs = torch.quasirandom.SobolEngine(dimension=4, scramble=True).draw(num_sample)
        else:
            raise NotImplementedError
        self.sample_visited = {i:0 for i in range(50000)}

    def __getitem__(self, index: int):
        if self.args.sobol_type == 'independent':
            x = self.sobolengs[index].draw()[0]
        elif self.args.sobol_type == 'overlap':
            x = self.sobolengs[index + self.sample_visited[index]]
        elif self.args.sobol_type == 'identical':
            x = self.sobolengs[self.sample_visited[index]]
        else:
            raise NotImplementedError

        (img,target) = self.cifar100.__getitem__(index)
        s = unit_interval_to_categorical(x[2], 2)
        i = unit_interval_to_categorical(x[0], 9)
        j = unit_interval_to_categorical(x[1], 9)
        angle = -15 + x[3] * 30
        if s != 0:
            img = F.hflip(img)
        img = F.pad(img, 4)
        img = F.crop(img, i, j, 32, 32)
        img = F.rotate(img, angle)
        if self.transform is not None:
            img = self.transform(img)
        self.state += 1
        if self.args.sobol_type in ['overlap', 'identical']:
            self.sample_visited[index] += 1
        return (img,target)

    def __len__(self) -> int:
        return len(self.cifar100.data)


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        # >>> transforms.Compose([
        # >>>     transforms.CenterCrop(10),
        # >>>     transforms.ToTensor(),
        # >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return F.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class QMCCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill: Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant
        padding_mode: Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.
             - constant: pads with a constant value, this value is specified with fill
             - edge: pads with the last value on the edge of the image
             - reflect: pads with reflection of image (without repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]
             - symmetric: pads with reflection of image (repeating the last value on the edge)
                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]
    """

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.index = 0
        self.seed = 0
        self.qmc_unit_samples = qp.Sobol(dimension=4, seed=self.seed).gen_samples(65536)


    def get_params(self, img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        # i = random.randint(0, h - th)
        # j = random.randint(0, w - tw)

        i = int(self.qmc_unit_samples[self.index][0]*(h - th)+0.5)
        j = int(self.qmc_unit_samples[self.index][1]*(w - tw)+0.5)
        if self.index == 65535:
            self.seed += 1
            self.qmc_unit_samples = qp.Sobol(dimension=4, seed=self.seed).gen_samples(65536)
            self.index = 0
        else:
            self.index += 1

        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class QMCHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p
        self.seed = 0
        self.qmc_unit_samples = qp.Sobol(dimension=4, seed=self.seed).gen_samples(65536)
        self.index = 0

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        self.p = self.qmc_unit_samples[self.index][2]
        if self.index == 65535:
            self.seed += 1
            self.qmc_unit_samples = qp.Sobol(1).gen_samples(65536)
            self.index = 0
        else:
            self.index += 1
        return self.__class__.__name__ + '(p={})'.format(self.p)
        # return self.__class__.__name__ + '(p={})'.format(self.p)

class QMCRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill
        self.index = 0
        self.seed = 0
        self.qmc_unit_samples = qp.Sobol(dimension=4, seed=self.seed).gen_samples(65536)

    def get_params(self, degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        # angle = random.uniform(degrees[0], degrees[1])

        angle = degrees[0] + self.qmc_unit_samples[self.index][3] * (degrees[1] - degrees[0])

        if self.index == 65535:
            self.seed += 1
            self.qmc_unit_samples = qp.Sobol(1).gen_samples(65536)
            self.index = 0
        else:
            self.index += 1

        return angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center, self.fill)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter. See `filters`_ for more information.
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
        fill (3-tuple or int): RGB pixel fill value for area outside the rotated image.
            If int, it is used for all channels respectively.
    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters
    """

    def __init__(self, degrees, resample=False, expand=False, center=None, fill=0):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.resample = resample
        self.expand = expand
        self.center = center
        self.fill = fill

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be rotated.
        Returns:
            PIL Image: Rotated image.
        """

        angle = self.get_params(self.degrees)

        return F.rotate(img, angle, self.resample, self.expand, self.center, self.fill)

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string
