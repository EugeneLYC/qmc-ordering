import torch
import json
from .transforms import QMCTransforms
from .constants import _TO_TENSOR_,  _PIL_TO_TENSOR_, _CONVERT_IMAGE_DTYPE_, _TO_PIL_IMAGE_, \
    _NORMALIZE_, _RESIZE_, _CENTER_CROP_, _PAD_, _LAMBDA_, _RANDOM_APPLY_, _RANDOM_CHOICE_, \
    _RANDOM_ORDER_, _RANDOM_CROP_, _RANDOM_HORIZONTAL_FLIP_, _RANDOM_VERTICAL_FLIP_, _RANDOM_RESIZED_CROP_, \
    _FIVE_CROP_, _TEN_CROP_, _LINEAR_TRANSFORMATION_, _COLOR_JITTER_, _RANDOM_ROTATION_, _RANDOM_AFFINE_, \
    _GRAY_SCALE_, _RANDOM_GRAY_SCALE_, _RANDOM_PERSPECTIVE_, _RANDOM_ERASING_, _GAUSSIAN_BLUR_, \
    _INTERPOLATION_MODE_, _RANDOM_INVERT_, _RANDOM_POSTERIZE_, _RANDOM_SOLARIZE_, _RANDOM_ADJUST_SHARPNESS_, \
    _RANDOM_AUTOCONTRAST_, _RANDOM_EQUALIZE_

def get_transforms(args):
    config = json.load(
        open(args.transforms_json, 'r', encoding='utf-8'))
    transforms = []
    qmc_quotas = []
    for key in config.keys():
        transform_params = config[key]
        if key == _TO_TENSOR_:
            from .transforms import ToTensor
            transforms.append(
                ToTensor())
            qmc_quotas.append(0)

        elif key == _RANDOM_HORIZONTAL_FLIP_:
            from .transforms import RandomHorizontalFlip
            transforms.append(
                RandomHorizontalFlip(**transform_params))
            qmc_quotas.append(1)

        elif key == _RANDOM_CROP_:
            from .transforms import RandomCrop
            transforms.append(
                RandomCrop(**transform_params))
            qmc_quotas.append(2)
        
        elif key == _NORMALIZE_:
            from .transforms import Normalize
            transforms.append(
                Normalize(**transform_params))
            qmc_quotas.append(0)
        
        else:
            raise NotImplementedError("This transform method has not been cleaned up yet.")
    
    qmc_transforms = QMCTransforms(
        transforms=transforms,
        qmc_quotas=qmc_quotas
    )
    
    return qmc_transforms, qmc_quotas
