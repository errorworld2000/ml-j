from mini_ml.utils.register import TRANSFORMS


@TRANSFORMS.register()
class Compose:
    """将多个变换组合在一起。"""

    def __init__(self, transforms_list):
        self.transforms = transforms_list

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
