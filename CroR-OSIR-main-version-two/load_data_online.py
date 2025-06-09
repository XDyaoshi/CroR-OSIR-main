# online data loader
import re
import os
import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
from torch.utils import data
import torchvision.transforms as transforms
'''This is a piece of instrumental code that defines the structure of the captured data'''
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def pil_loader(path, mode):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            if mode == "L":
                return img.convert('L')
            if mode == "RGB":
                return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path, mode="RGB"):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path, mode)


def is_image_file(filename):
    """Checks if a file is an image.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)


def find_classes(dirs):
    classes = [] 
    for dir in dirs:
        classes.extend([d for d in os.listdir(dir)])
    for i in range(len(classes)):
        lis = os.listdir(dir)
        str = lis[0][-4:]
        classes[i] = classes[i].replace(str,"")
        classes[i] = int(classes[i])
    classes.sort()

    class_to_idx = {i: i for i in classes}
    return classes, class_to_idx

'''The purpose of this function is to return a list of tuples containing the image path and its category label'''
def make_dataset(dirs, class_to_idx):

    images = []
    for dir in dirs:
        dir = os.path.expanduser(dir)
        for target in sorted(os.listdir(dir)):
            image_last_name = target[-4:]
            target_n = target.replace(image_last_name,"")
            y = int(target_n)
            d = os.path.join(dir, target) #This function is responsible for concatenating paths
            d = os.path.join(dir)
            if not os.path.isdir(d):
                continue
            count = 0
            for root, _, fnames in sorted(os.walk(d)): 
                
                my_string = root
                match = re.search(r'\d', my_string)  
                if match:
                    index_of_digit = match.start() 
                    substring_until_digit = my_string[:index_of_digit]  
                for fname in sorted(fnames): 
                    # Please do not include numbers in the dataset folder name
                    if is_image_file(fname):
                        path = os.path.join(root, fname)
                        class_ids = root.replace(substring_until_digit,"") 
                        class_ids = class_ids.replace("/","")
                        item = (path, int(class_ids))
                        images.append(item)
                count = count+1
            if (count!=0):
                break
            else:
                continue
    return images


class CostumeImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, roots, transform=None, target_transform=None,
                 loader=default_loader, mode="RGB"):
        classes, class_to_idx = find_classes(roots)
        imgs = make_dataset(roots, class_to_idx)
    
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders",
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.roots = roots
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.mode = mode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path, self.mode)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class CostumeMixedImageFolder(CostumeImageFolder):
    def __init__(self, roots, category_indexs, transform, target_transform=None,
                 loader=default_loader, mode="RGB"):
        classes, class_to_idx = find_classes(roots)
        imgs = make_dataset(roots, class_to_idx)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders",
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.roots = roots
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.mode = mode
        self.labeled_transform = self.transform[0]
        self.unlabeled_transform = self.transform[1]
        self.category_indexs = category_indexs

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path, self.mode)
        if self.transform is not None:
            if target in self.category_indexs:
                img = self.labeled_transform(img)
            else:
                img = self.unlabeled_transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target