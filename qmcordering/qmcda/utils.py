import json
from .transforms import QMCTransforms
from .constants import _TO_TENSOR_,  _PIL_TO_TENSOR_, _CONVERT_IMAGE_DTYPE_, _TO_PIL_IMAGE_, \
    _NORMALIZE_, _RESIZE_, _CENTER_CROP_, _PAD_, _LAMBDA_, _RANDOM_APPLY_, _RANDOM_CHOICE_, \
    _RANDOM_ORDER_, _RANDOM_CROP_, _RANDOM_HORIZONTAL_FLIP_, _RANDOM_VERTICAL_FLIP_, _RANDOM_RESIZED_CROP_, \
    _FIVE_CROP_, _TEN_CROP_, _LINEAR_TRANSFORMATION_, _COLOR_JITTER_, _RANDOM_ROTATION_, _RANDOM_AFFINE_, \
    _GRAY_SCALE_, _RANDOM_GRAY_SCALE_, _RANDOM_PERSPECTIVE_, _RANDOM_ERASING_, _GAUSSIAN_BLUR_, \
    _RANDOM_INVERT_, _RANDOM_POSTERIZE_, _RANDOM_SOLARIZE_, _RANDOM_ADJUST_SHARPNESS_, \
    _RANDOM_AUTOCONTRAST_, _RANDOM_EQUALIZE_

def get_qmc_transforms(config):
    transforms = []
    qmc_quotas = []
    for key in config.keys():
        transform_params = config[key]
        if key == _TO_TENSOR_:
            from .transforms import ToTensor
            transforms.append(
                ToTensor())
            qmc_quotas.append(0)
        
        elif key == _PIL_TO_TENSOR_:
            from .transforms import PILToTensor
            transforms.append(
                PILToTensor())
            qmc_quotas.append(0)
        
        elif key == _CONVERT_IMAGE_DTYPE_:
            from .transforms import ConvertImageDtype
            transforms.append(
                ConvertImageDtype(**transform_params))
            qmc_quotas.append(0)
        
        elif key == _TO_PIL_IMAGE_:
            from .transforms import ToPILImage
            transforms.append(
                ToPILImage(**transform_params))
            qmc_quotas.append(0)
        
        elif key == _NORMALIZE_:
            from .transforms import Normalize
            transforms.append(
                Normalize(**transform_params))
            qmc_quotas.append(0)
        
        elif key == _RESIZE_:
            from .transforms import Resize
            transforms.append(
                Resize(**transform_params))
            qmc_quotas.append(0)
        
        elif key == _CENTER_CROP_:
            from .transforms import CenterCrop
            transforms.append(
                CenterCrop(**transform_params))
            qmc_quotas.append(0)
        
        elif key == _PAD_:
            from .transforms import Pad
            transforms.append(
                Pad(**transform_params))
            qmc_quotas.append(0)
        
        elif key == _LAMBDA_:
            from .transforms import Lambda
            transforms.append(
                Lambda(**transform_params))
            qmc_quotas.append(0)
        
        elif key == _RANDOM_APPLY_:
            from .transforms import RandomApply
            transforms.append(
                RandomApply(**transform_params))
            qmc_quotas.append(0)
        
        elif key == _RANDOM_ORDER_:
            from .transforms import RandomOrder
            transforms.append(
                RandomOrder())
            qmc_quotas.append(0)
        
        elif key == _RANDOM_CHOICE_:
            from .transforms import RandomChoice
            transforms.append(
                RandomChoice(**transform_params))
            qmc_quotas.append(0)
        
        elif key == _RANDOM_CROP_:
            from .transforms import RandomCrop
            transforms.append(
                RandomCrop(**transform_params))
            qmc_quotas.append(2)

        elif key == _RANDOM_HORIZONTAL_FLIP_:
            from .transforms import RandomHorizontalFlip
            transforms.append(
                RandomHorizontalFlip(**transform_params))
            qmc_quotas.append(1)
        
        elif key == _RANDOM_VERTICAL_FLIP_:
            from .transforms import RandomVerticalFlip
            transforms.append(
                RandomVerticalFlip(**transform_params))
            qmc_quotas.append(1)
        
        elif key == _RANDOM_PERSPECTIVE_:
            from .transforms import RandomPerspective
            transforms.append(
                RandomPerspective(**transform_params))
            qmc_quotas.append(1)
        
        elif key == _RANDOM_RESIZED_CROP_:
            from .transforms import RandomResizedCrop
            transforms.append(
                RandomResizedCrop(**transform_params))
            qmc_quotas.append(4)
        
        elif key == _FIVE_CROP_:
            from .transforms import FiveCrop
            transforms.append(
                FiveCrop(**transform_params))
            qmc_quotas.append(0)
        
        elif key == _TEN_CROP_:
            from .transforms import TenCrop
            transforms.append(
                TenCrop(**transform_params))
            qmc_quotas.append(0)
        
        elif key == _LINEAR_TRANSFORMATION_:
            from .transforms import LinearTransformation
            transforms.append(
                LinearTransformation(**transform_params))
            qmc_quotas.append(0)
        
        elif key == _COLOR_JITTER_:
            from .transforms import ColorJitter
            transforms.append(
                ColorJitter(**transform_params))
            qmc_quotas.append(4)
        
        elif key == _RANDOM_ROTATION_:
            from .transforms import RandomRotation
            transforms.append(
                RandomRotation(**transform_params))
            qmc_quotas.append(1)
        
        elif key == _RANDOM_AFFINE_:
            from .transforms import RandomAffine
            transforms.append(
                RandomAffine(**transform_params))
            qmc_quotas.append(4)
        
        elif key == _GRAY_SCALE_:
            from .transforms import Grayscale
            transforms.append(
                Grayscale(**transform_params))
            qmc_quotas.append(0)
        
        elif key == _RANDOM_GRAY_SCALE_:
            from .transforms import RandomGrayscale
            transforms.append(
                RandomGrayscale(**transform_params))
            qmc_quotas.append(1)
        
        elif key == _RANDOM_ERASING_:
            from .transforms import RandomErasing
            transforms.append(
                RandomErasing(**transform_params))
            qmc_quotas.append(4)
        
        elif key == _GAUSSIAN_BLUR_:
            from .transforms import GaussianBlur
            transforms.append(
                GaussianBlur(**transform_params))
            qmc_quotas.append(1)
        
        elif key == _RANDOM_INVERT_:
            from .transforms import RandomInvert
            transforms.append(
                RandomInvert(**transform_params))
            qmc_quotas.append(1)
        
        elif key == _RANDOM_POSTERIZE_:
            from .transforms import RandomPosterize
            transforms.append(
                RandomPosterize(**transform_params))
            qmc_quotas.append(1)
        
        elif key == _RANDOM_SOLARIZE_:
            from .transforms import RandomSolarize
            transforms.append(
                RandomSolarize(**transform_params))
            qmc_quotas.append(1)
        
        elif key == _RANDOM_ADJUST_SHARPNESS_:
            from .transforms import RandomAdjustSharpness
            transforms.append(
                RandomAdjustSharpness(**transform_params))
            qmc_quotas.append(1)
        
        elif key == _RANDOM_AUTOCONTRAST_:
            from .transforms import RandomAutocontrast
            transforms.append(
                RandomAutocontrast(**transform_params))
            qmc_quotas.append(1)
        
        elif key == _RANDOM_EQUALIZE_:
            from .transforms import RandomEqualize
            transforms.append(
                RandomEqualize(**transform_params))
            qmc_quotas.append(1)
        
        else:
            raise NotImplementedError("This transform method: {} is not supported.".format(key))
    
    qmc_transforms = QMCTransforms(
        transforms=transforms,
        qmc_quotas=qmc_quotas
    )
    
    return qmc_transforms, qmc_quotas

def get_uniform_transforms(config):
    transforms = []
    for key in config.keys():
        transform_params = config[key]
        if key == _TO_TENSOR_:
            from torchvision.transforms import ToTensor
            transforms.append(
                ToTensor())
        
        elif key == _PIL_TO_TENSOR_:
            from torchvision.transforms import PILToTensor
            transforms.append(
                PILToTensor())
        
        elif key == _CONVERT_IMAGE_DTYPE_:
            from torchvision.transforms import ConvertImageDtype
            transforms.append(
                ConvertImageDtype(**transform_params))
        
        elif key == _TO_PIL_IMAGE_:
            from torchvision.transforms import ToPILImage
            transforms.append(
                ToPILImage(**transform_params))
        
        elif key == _NORMALIZE_:
            from torchvision.transforms import Normalize
            transforms.append(
                Normalize(**transform_params))
        
        elif key == _RESIZE_:
            from torchvision.transforms import Resize
            transforms.append(
                Resize(**transform_params))
        
        elif key == _CENTER_CROP_:
            from torchvision.transforms import CenterCrop
            transforms.append(
                CenterCrop(**transform_params))
        
        elif key == _PAD_:
            from torchvision.transforms import Pad
            transforms.append(
                Pad(**transform_params))
        
        elif key == _LAMBDA_:
            from torchvision.transforms import Lambda
            transforms.append(
                Lambda(**transform_params))
        
        elif key == _RANDOM_APPLY_:
            from torchvision.transforms import RandomApply
            transforms.append(
                RandomApply(**transform_params))
        
        elif key == _RANDOM_ORDER_:
            from torchvision.transforms import RandomOrder
            transforms.append(
                RandomOrder())
        
        elif key == _RANDOM_CHOICE_:
            from torchvision.transforms import RandomChoice
            transforms.append(
                RandomChoice(**transform_params))
        
        elif key == _RANDOM_CROP_:
            from torchvision.transforms import RandomCrop
            transforms.append(
                RandomCrop(**transform_params))

        elif key == _RANDOM_HORIZONTAL_FLIP_:
            from torchvision.transforms import RandomHorizontalFlip
            transforms.append(
                RandomHorizontalFlip(**transform_params))
        
        elif key == _RANDOM_VERTICAL_FLIP_:
            from torchvision.transforms import RandomVerticalFlip
            transforms.append(
                RandomVerticalFlip(**transform_params))
        
        elif key == _RANDOM_PERSPECTIVE_:
            from torchvision.transforms import RandomPerspective
            transforms.append(
                RandomPerspective(**transform_params))
        
        elif key == _RANDOM_RESIZED_CROP_:
            from torchvision.transforms import RandomResizedCrop
            transforms.append(
                RandomResizedCrop(**transform_params))
        
        elif key == _FIVE_CROP_:
            from torchvision.transforms import FiveCrop
            transforms.append(
                FiveCrop(**transform_params))
        
        elif key == _TEN_CROP_:
            from torchvision.transforms import TenCrop
            transforms.append(
                TenCrop(**transform_params))
        
        elif key == _LINEAR_TRANSFORMATION_:
            from torchvision.transforms import LinearTransformation
            transforms.append(
                LinearTransformation(**transform_params))
        
        elif key == _COLOR_JITTER_:
            from torchvision.transforms import ColorJitter
            transforms.append(
                ColorJitter(**transform_params))
        
        elif key == _RANDOM_ROTATION_:
            from torchvision.transforms import RandomRotation
            transforms.append(
                RandomRotation(**transform_params))
        
        elif key == _RANDOM_AFFINE_:
            from torchvision.transforms import RandomAffine
            transforms.append(
                RandomAffine(**transform_params))
        
        elif key == _GRAY_SCALE_:
            from torchvision.transforms import Grayscale
            transforms.append(
                Grayscale(**transform_params))
        
        elif key == _RANDOM_GRAY_SCALE_:
            from torchvision.transforms import RandomGrayscale
            transforms.append(
                RandomGrayscale(**transform_params))
        
        elif key == _RANDOM_ERASING_:
            from torchvision.transforms import RandomErasing
            transforms.append(
                RandomErasing(**transform_params))
        
        elif key == _GAUSSIAN_BLUR_:
            from torchvision.transforms import GaussianBlur
            transforms.append(
                GaussianBlur(**transform_params))
        
        elif key == _RANDOM_INVERT_:
            from torchvision.transforms import RandomInvert
            transforms.append(
                RandomInvert(**transform_params))
        
        elif key == _RANDOM_POSTERIZE_:
            from torchvision.transforms import RandomPosterize
            transforms.append(
                RandomPosterize(**transform_params))
        
        elif key == _RANDOM_SOLARIZE_:
            from torchvision.transforms import RandomSolarize
            transforms.append(
                RandomSolarize(**transform_params))
        
        elif key == _RANDOM_ADJUST_SHARPNESS_:
            from torchvision.transforms import RandomAdjustSharpness
            transforms.append(
                RandomAdjustSharpness(**transform_params))
        
        elif key == _RANDOM_AUTOCONTRAST_:
            from torchvision.transforms import RandomAutocontrast
            transforms.append(
                RandomAutocontrast(**transform_params))
        
        elif key == _RANDOM_EQUALIZE_:
            from torchvision.transforms import RandomEqualize
            transforms.append(
                RandomEqualize(**transform_params))
        
        else:
            raise NotImplementedError("This transform method: {} is not supported.".format(key))
    
    transforms = torchvision.transforms.Compose(transforms)
    
    return transforms
