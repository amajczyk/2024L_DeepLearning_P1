import albumentations as A
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


class CustomAugmentations:
    '''
    This class is used to apply custom augmentations to the images using the albumentations library
    Parameters:
    image: np.ndarray
        The image to be augmented
    p_dict: dict
        The dictionary containing the probability of each augmentation to be applied
    '''
    def __init__(self, p_dict):
        self.p_dict = p_dict
        self.mapping = {
            "flip": A.Flip,
            "transpose": A.Transpose,
            "gauss_noise": A.GaussNoise,
            "blur": A.OneOf,
            "shift_scale_rotate": A.ShiftScaleRotate,
            "distortion": A.OneOf,
            "brightness_contrast": A.OneOf,
            "hue_saturation_value": A.HueSaturationValue,
            "perspective": A.Perspective,
            "rotate": A.Rotate
        }
        if not self.__check_p_dict__():
            raise ValueError("The p_dict is not valid. Please check the values")
    
        transform = []
        for k, v in self.p_dict.items():
            if v > 0:
                if k == "blur":
                    transform.append(self.mapping[k]([
                        A.MotionBlur(p=1),
                        A.MedianBlur(blur_limit=3, p=1),
                        A.Blur(blur_limit=3, p=1)
                    ], p=v))
                elif k == "distortion":
                    transform.append(self.mapping[k]([
                        A.OpticalDistortion(p=1),
                        A.GridDistortion(p=1)
                    ], p=v))
                elif k == "brightness_contrast":
                    transform.append(self.mapping[k]([
                        A.CLAHE(clip_limit=2, p=1),
                        A.RandomBrightnessContrast(p=1)
                    ], p=v))
                else:
                    transform.append(self.mapping[k](p=v))
        self.transform = A.Compose(transform)
    
    def __check_p_dict__(self):
        '''
        This method is used to check if the p_dict is valid

        Returns:
        bool
            True if the p_dict is valid, False otherwise
        '''
        for k, v in self.p_dict.items():
            if v < 0 or v > 1 or k not in self.mapping.keys():
                return False
        return True
    
    def augment(self):
        '''
        This method is used to apply the augmentations to the image

        Returns:
        np.ndarray
            The augmented image
        '''
        transform = self.__transform__()
        return transform(image=self.image)['image']