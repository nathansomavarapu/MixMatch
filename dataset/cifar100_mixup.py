import numpy as np
from PIL import Image

import torchvision
import torch
import os
import pickle

supercatagories = {
    'bever'             : 'aquatic_mammals',
    'dolphin'           : 'aquatic_mammals',
    'otter'             : 'aquatic_mammals',
    'seal'              : 'aquatic_mammals',
    'whale'             : 'aquatic_mammals',
    'aquarium_fish'     : 'fish',
    'flatfish'          : 'fish',
    'ray'               : 'fish',
    'shark'             : 'fish',
    'trout'             : 'fish',
    'orchids'           : 'flowers',
    'poppies'           : 'flowers',
    'roses'             : 'flowers',
    'sunflowers'        : 'flowers',
    'tulips'            : 'flowers',
    'bottles'           : 'food_containers',
    'bowls'             : 'food_containers',
    'cans'              : 'food_containers',
    'cups'              : 'food_containers',
    'plates'            : 'food_containers',
    'apples'            : 'fruit_and_vegetables',
    'mushrooms'         : 'fruit_and_vegetables',
    'oranges'           : 'fruit_and_vegetables',
    'pears'             : 'fruit_and_vegetables',
    'sweet_peppers'     : 'fruit_and_vegetables',
    'clock'             : 'household_electrical_devices',
    'computer_keyboard' : 'household_electrical_devices',
    'lamp'              : 'household_electrical_devices',
    'telephone'         : 'household_electrical_devices',
    'television'        : 'household_electrical_devices',
    'bed'               : 'household_furniture',
    'chair'             : 'household_furniture',
    'couch'             : 'household_furniture',
    'table'             : 'household_furniture',
    'wardrobe'          : 'household_furniture',
    'bee'               : 'insects',
    'beetle'            : 'insects',
    'butterfly'         : 'insects',
    'caterpillar'       : 'insects',
    'cockroach'         : 'insects',
    'bear'              : 'large_carnivores',
    'leopard'           : 'large_carnivores',
    'lion'              : 'large_carnivores',
    'tiger'             : 'large_carnivores',
    'wolf'              : 'large_carnivores',
    'bridge'            : 'large_man-made_outdoor_things',
    'castle'            : 'large_man-made_outdoor_things',
    'house'             : 'large_man-made_outdoor_things',
    'road'              : 'large_man-made_outdoor_things',
    'skyscraper'        : 'large_man-made_outdoor_things',
    'cloud'             : 'large_natural_outdoor_scenes',
    'forest'            : 'large_natural_outdoor_scenes',
    'mountain'          : 'large_natural_outdoor_scenes',
    'plain'             : 'large_natural_outdoor_scenes',
    'sea'               : 'large_natural_outdoor_scenes',
    'camel'             : 'large_omnivores_and_herbivores',
    'cattle'            : 'large_omnivores_and_herbivores',
    'chimpanzee'        : 'large_omnivores_and_herbivores',
    'elephant'          : 'large_omnivores_and_herbivores',
    'kangaroo'          : 'large_omnivores_and_herbivores',
    'fox'               : 'medium-sized_mammals',
    'porcupine'         : 'medium-sized_mammals',
    'possum'            : 'medium-sized_mammals',
    'raccoon'           : 'medium-sized_mammals',
    'skunk'             : 'medium-sized_mammals',
    'crab'              : 'non-insect_invertebrates',
    'lobster'           : 'non-insect_invertebrates',
    'snail'             : 'non-insect_invertebrates',
    'spider'            : 'non-insect_invertebrates',
    'worm'              : 'non-insect_invertebrates',
    'baby'              : 'people',
    'boy'               : 'people',
    'girl'              : 'people',
    'man'               : 'people',
    'woman'             : 'people',
    'crocodile'         : 'reptiles',
    'dinosaur'          : 'reptiles',
    'lizard'            : 'reptiles',
    'snake'             : 'reptiles',
    'turtle'            : 'reptiles',
    'hamster'           : 'small_mammals',
    'mouse'             : 'small_mammals',
    'rabbit'            : 'small_mammals',
    'shrew'             : 'small_mammals',
    'squirrel'          : 'small_mammals',
    'maple'             : 'trees',
    'oak'               : 'trees',
    'palm'              : 'trees',
    'pine'              : 'trees',
    'willow'            : 'trees',
    'bicycle'           : 'vehicles_1',
    'bus'               : 'vehicles_1',
    'motorcycle'        : 'vehicles_1',
    'pickup_truck'      : 'vehicles_1',
    'train'             : 'vehicles_1',
    'lawn-mower'        : 'vehicles_2',
    'rocket'            : 'vehicles_2',
    'streetcar'         : 'vehicles_2',
    'tank'              : 'vehicles_2',
    'tractor'           : 'vehicles_2',
}

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2

def get_cifar100(root, n_labeled,
                 transform_train=None, transform_val=None,
                 download=True, debug=False):

    base_dataset = torchvision.datasets.CIFAR100(root, train=True, download=download)
    train_labeled_idxs, train_unlabeled_idxs, val_idxs = train_val_split(base_dataset.targets, int(n_labeled/100))

    train_labeled_dataset = CIFAR100_labeled(root, train_labeled_idxs, train=True, transform=transform_train)
    train_unlabeled_dataset = CIFAR100_unlabeled(root, train_unlabeled_idxs, train=True, transform=TransformTwice(transform_train), debug=debug)
    val_dataset = CIFAR100_labeled(root, val_idxs, train=True, transform=transform_val, download=True)
    test_dataset = CIFAR100_labeled(root, train=False, transform=transform_val, download=True)

    print (f"#Labeled: {len(train_labeled_idxs)} #Unlabeled: {len(train_unlabeled_idxs)} #Val: {len(val_idxs)}")
    return train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset
    

def train_val_split(labels, n_labeled_per_class, val_per_class=200):
    labels = np.array(labels)
    train_labeled_idxs = []
    train_unlabeled_idxs = []
    val_idxs = []

    for i in range(100):
        idxs = np.where(labels == i)[0]
        np.random.shuffle(idxs)
        train_labeled_idxs.extend(idxs[:n_labeled_per_class])
        train_unlabeled_idxs.extend(idxs[n_labeled_per_class:-val_per_class])
        val_idxs.extend(idxs[-val_per_class:])
    np.random.shuffle(train_labeled_idxs)
    np.random.shuffle(train_unlabeled_idxs)
    np.random.shuffle(val_idxs)

    return train_labeled_idxs, train_unlabeled_idxs, val_idxs

cifar100_mean = (0.4914, 0.4822, 0.4465) # equals np.mean(train_set.train_data, axis=(0,1,2))/255
cifar100_std = (0.2471, 0.2435, 0.2616) # equals np.std(train_set.train_data, axis=(0,1,2))/255

def normalise(x, mean=cifar100_mean, std=cifar100_std):
    x, mean, std = [np.array(a, np.float32) for a in (x, mean, std)]
    x -= mean*255
    x *= 1.0/(255*std)
    return x

def transpose(x, source='NHWC', target='NCHW'):
    return x.transpose([source.index(d) for d in target]) 

def pad(x, border=4):
    return np.pad(x, [(0, 0), (border, border), (border, border)], mode='reflect')

class RandomPadandCrop(object):
    """Crop randomly the image.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, x):
        x = pad(x, 4)

        h, w = x.shape[1:]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        x = x[:, top: top + new_h, left: left + new_w]

        return x

class RandomFlip(object):
    """Flip randomly the image.
    """
    def __call__(self, x):
        if np.random.rand() < 0.5:
            x = x[:, :, ::-1]

        return x.copy()

class GaussianNoise(object):
    """Add gaussian noise to the image.
    """
    def __call__(self, x):
        c, h, w = x.shape
        x += np.random.randn(c, h, w) * 0.15
        return x

class ToTensor(object):
    """Transform the image to tensor.
    """
    def __call__(self, x):
        x = torch.from_numpy(x)
        return x

class CIFAR100_labeled(torchvision.datasets.CIFAR100):

    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_labeled, self).__init__(root, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        
        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list
        
        self.coarse_targets = []
        
        for file_name, checksum in downloaded_list:
            fp = os.path.join(self.root, self.base_folder, file_name)
            with open(fp, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                if 'coarse_labels' in entry:
                    self.coarse_targets.extend(entry['coarse_labels'])
        
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]
            self.coarse_targets = np.array(self.coarse_targets)[indexs]
        
        supercatagories = torch.unique(torch.Tensor(self.coarse_targets))
        self.data_idxs = []
        for cat in supercatagories:
            it = iter((torch.Tensor(self.coarse_targets) == cat).byte().nonzero().numpy())
            self.data_idxs.extend(zip(it,it))
        
        self.data = transpose(normalise(self.data))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        idx1, idx2 = self.data_idxs[index]

        img1, target1 = self.data[idx1], self.targets[idx1]
        img2, target2 = self.data[idx2], self.targets[idx2]
        
        print(type(img1))
        exit()

        if self.transform is not None:
            img1 = self.transform(img1).unsqueeze(0)
            img2 = self.transform(img2).unsqueeze(0)

        if self.target_transform is not None:
            target1 = self.target_transform(target1).unsqueeze(0)
            target2 = self.target_transform(target2).unsqueeze(0)

        return torch.cat([img1,img2]), torch.cat([target1, target2])
    
    def __len__(self):
        return len(self.data_idxs)
    

class CIFAR100_unlabeled(CIFAR100_labeled):

    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False, debug=False):
        super(CIFAR100_unlabeled, self).__init__(root, indexs, train=train,
                 transform=transform, target_transform=target_transform,
                 download=download)
        
        # Debug keeps the unsupervised classes for oracle comparisons
        if not debug:
            self.targets = np.array([-1 for i in range(len(self.targets))])

def collate_mixup(batch):
    torch.cat(batch)
        