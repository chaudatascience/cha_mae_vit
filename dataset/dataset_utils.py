from torch import Tensor


class self_normalize(object):
    def __call__(self, x):
        m = x.mean((-2, -1), keepdim=True)
        s = x.std((-2, -1), unbiased=False, keepdim=True)
        x -= m
        x /= s + 1e-7
        return x


do_nothing = lambda x: x


class TorchvisionWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, *args, **kwargs):
        out = self.transform(image)
        return out


class Regular2Alubmentation:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, *args, **kwargs):
        out = self.transform(image, *args, **kwargs)
        return {"image": out}


class AlubmentationWrapper:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, image, *args, **kwargs):
        out = self.transform(image=image, *args, **kwargs)
        return out["image"]


"""
dataset = file_group[0].split(os.sep)[1]
baseline = get_crop_size(dataset)

if baseline != (-1, -1):
    crop_height = random.randint(int(baseline[0] * 0.9), int(baseline[0] * 1.1))
    crop_width = random.randint(int(baseline[1] * 0.9), int(baseline[1] * 1.1))
    self.guided_crops.crop_size = (crop_height, crop_width)
else:
    self.guided_crops.crop_size = (-1, -1)

if self.guided_crops.crop_size != (-1, -1) and self.config.guided_crops_path:
    safetensors_name: str = file_group[0][:-4] + ".safetensors"
    safetensors_name = "CHAMMI-75_guidance/" + safetensors_name.split("/", maxsplit=1)[1]
    if safetensors_name in self.guided_crops.data_paths:
        image_tensor = self.guided_crops(image_tensor, safetensors_name)
print(f"image_tensor after: {image_tensor.shape}, channel_types after: {channel_types}")
"""
