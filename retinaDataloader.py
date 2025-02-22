import torch.utils.data as data
from PIL import Image,ImageEnhance
import os
import torchvision.transforms as transforms
import torch
import numpy as np
import random
import torchvision.transforms.functional as F
import torch.nn.functional as Funct



class RetinaDataset(data.Dataset):
    def __init__(self, data_path, names, _augment):
        super(RetinaDataset, self).__init__()
        self.data_path = data_path
        self.names = names
        self.augment = _augment

    @staticmethod
    def augmentate(image, mask, entropy):


        # it is expected to be in [..., H, W] format
        image = torch.from_numpy(
            np.array(image, dtype=np.uint8)).permute(2, 0, 1)
        mask = torch.unsqueeze(torch.from_numpy(
            np.array(mask, dtype=np.uint8)), dim=0)
        entropy = torch.from_numpy(
            np.array(entropy, dtype=np.uint8)).permute(2, 0, 1)
        entropy = Funct.interpolate(entropy.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False)
        entropy = entropy.squeeze(0)

        entropy = entropy[:3, :, :]  # Keep only the first 3 channels (RGB)

        # print(entropy.shape) #torch.Size([512, 3, 512])
        # print(image.shape) #torch.Size([512, 3, 512])
        image = F.adjust_gamma(image, gamma=random.uniform(0.8, 1.2))
        image = F.adjust_contrast(
            image, contrast_factor=random.uniform(0.8, 1.2))
        image = F.adjust_brightness(
            image, brightness_factor=random.uniform(0.8, 1.2))
        image = F.adjust_saturation(
            image, saturation_factor=random.uniform(0.8, 1.2))
        image = F.adjust_hue(image, hue_factor=random.uniform(-0.2, 0.2))



        image_mask_entropy = torch.cat([image, mask, entropy], dim=0)
        # image_mask = torch.cat([image, mask], dim=0)


        if random.uniform(0, 1) > 0.5:
            image_mask_entropy = F.hflip(image_mask_entropy)
        if random.uniform(0, 1) > 0.5:
            image_mask_entropy = F.vflip(image_mask_entropy)
        if random.uniform(0, 1) > 0.5:
            image_mask_entropy = F.rotate(image_mask_entropy, angle=90)
        #

        image = image_mask_entropy[:3, ...]
        # Extract mask (1 channel)
        mask = image_mask_entropy[3:4, ...]  # Keep it as [1, H, W], no need to unsqueeze
        # Extract entropy (last 4 channels)
        entropy = image_mask_entropy[4:, ...]  # Fix this to get all entropy channels


        return F.to_pil_image(image), F.to_pil_image(mask), F.to_pil_image(entropy)

    def __getitem__(self, index):
        name = self.names[index]
        img_transform = transforms.Compose([
            transforms.Resize([256, 256]),
            # transforms.CenterCrop((256, 256)),
            transforms.ToTensor()
        ])
        retina = Image.open(os.path.join(self.data_path, "image", name))
        mask = Image.open(os.path.join(self.data_path, "mask", name)).convert('L')
        entropy = Image.open(os.path.join(self.data_path, "entropy", name))

        if self.augment:
            retina, mask, entropy = self.augmentate(retina, mask, entropy)

        retina = img_transform(retina)
        entropy = img_transform(entropy)
        mask = img_transform(mask)  # [1, h, w]

        return {
            "name": name,
            "retina": retina,
            "entropy": entropy,
            "mask": mask
        }

    def __len__(self):
        return len(self.names)


if __name__ == "__main__":
    with open(os.path.join("./data/retina500", "train.txt"), 'r') as f:
        train_areas = [line.split()[0] for line in f.readlines()]
    train_dataset = RetinaDataset("./data/retina500", train_areas, _augment=True)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=4,
                                               num_workers=4
    )
    c = next(iter(train_loader))["retina"]
    m = next(iter(train_loader))["mask"]
    print(c.shape, m.shape)