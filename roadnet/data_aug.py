import albumentations as A


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define heavy augmentations
def get_training_augmentation(height=352, width=1216):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=(-0.5, 2), rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.PadIfNeeded(min_height=height, min_width=width, always_apply=True, border_mode=0),
        A.RandomCrop(height=height, width=width, always_apply=True),
        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),
        A.OneOf([
            A.CLAHE(p=1),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
            A.RandomGamma(p=1),],
            p=0.9,
        ),

        A.OneOf([
            A.IAASharpen(p=1),
            A.Blur(blur_limit=3, p=1),
            A.MotionBlur(blur_limit=3, p=1),],
            p=0.9,
        ),

        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=1),
            A.HueSaturationValue(p=1),],
            p=0.9,
        ),
        A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(height=352, width=1216):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.PadIfNeeded(min_height=height, min_width=width)
    ]
    return A.Compose(test_transform)


def image_normalization(image, **kwargs):
    return image/255


def get_preprocessing():
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = [
        A.Lambda(image=image_normalization),
    ]
    return A.Compose(_transform)

