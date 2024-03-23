import tensorflow as tf
import numpy as np

class CustomAugmentationsTF:
    '''
    This class is used to apply custom augmentations to the images using TensorFlow
    Parameters:
    p_dict: dict
        The dictionary containing the probability of each augmentation to be applied
    '''
    def __init__(self, p_dict):
        self.p_dict = p_dict
    
    def augment(self, image):
        '''
        This method is used to apply the augmentations to the image

        Returns:
        tf.Tensor
            The augmented image
        '''
        for k, v in self.p_dict.items():
            if v > 0:
                image = self.apply_augmentation(image, k, v)
        return image

    def apply_augmentation(self, image, augmentation, probability):
        '''
        This method applies a specific augmentation to the image

        Parameters:
        image: tf.Tensor
            The input image
        augmentation: str
            The name of the augmentation to apply
        probability: float
            The probability of applying the augmentation

        Returns:
        tf.Tensor
            The augmented image
        '''
        random_number = tf.convert_to_tensor(np.random.uniform(0, 1), dtype=tf.float32)  # Convert numpy array to TensorFlow tensor
        
        if augmentation == "flip":
            return tf.image.flip_left_right(image) if random_number < probability else image
        elif augmentation == "transpose":
            return tf.image.transpose(image) if random_number < probability else image
        elif augmentation == "gauss_noise":
            noise = tf.random.normal(tf.shape(image), stddev=0.1)
            return image + noise if random_number < probability else image
        elif augmentation == "brightness_contrast":
            return tf.cond(random_number < probability,
                           lambda: tf.image.random_brightness(tf.image.random_contrast(image, 0.2, 0.5), 0.2),
                           lambda: image)
        elif augmentation == "hue_saturation_value":
            return tf.cond(random_number < probability,
                           lambda: tf.image.random_hue(tf.image.random_saturation(tf.image.random_brightness(image, 0.2), 0.2, 0.5), 0.2),
                           lambda: image)
        else:
            return image