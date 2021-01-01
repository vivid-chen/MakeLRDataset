import torch.utils.data as data

from os import listdir
from os.path import join
from PIL import Image, ImageFilter


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')

    #y, _, _ = img.split()
    #return y
    return img

class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        a = self.image_filenames[index]
        str_list = list(a)
        str_list.insert(27, "_lr")
        b = ''.join(str_list)
        input = load_img(self.image_filenames[index])
        target = load_img(b)
        if self.input_transform:
            # input = input.filter(ImageFilter.GaussianBlur(2))

            input = self.input_transform(input)
        if self.target_transform:
            target = self.target_transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenames)
