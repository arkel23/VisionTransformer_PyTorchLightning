import os

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder

from pytorch_lightning import LightningDataModule
from timm.data import create_transform

def return_prepared_dm(args):

    assert args.dataset_path, "Dataset path must not be empty."
   # setup data
    if args.dataset_name == 'cifar10':
        dm = CIFAR10DM(args)
    elif args.dataset_name == 'cifar100':
        dm = CIFAR100DM(args)
    elif args.dataset_name == 'imagenet':
        dm = ImageNetDM(args)
    
    dm.prepare_data()
    dm.setup('fit')
    args.num_classes = dm.num_classes
    
    return dm


def build_deit_transform(is_train, args):
    resize_im = args.image_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.image_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.image_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.image_size) # to maintain same ratio w.r.t. 224 images
        t.append(transforms.Resize(size, interpolation=3))
        t.append(transforms.CenterCrop(args.image_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225]))
    return transforms.Compose(t)


def get_transform(split, args):

    if split == 'train':
        if args.deit_recipe:
            transform = build_deit_transform(is_train=True, args=args)
        else:
            transform = transforms.Compose([
                transforms.Resize((args.image_size+32, args.image_size+32)),
                transforms.RandomCrop((args.image_size, args.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, 
                    contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                ])
    else:
        if args.deit_recipe:
            transform = build_deit_transform(is_train=False, args=args)
        else:
            transform = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)), 
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                ])

    return transform


class CIFAR10DM(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.data_dir = args.dataset_path
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.num_workers = args.no_cpu_workers
        self.transform_train = get_transform(split='train', args=args)
        self.transform_eval = get_transform(split='val', args=args)

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        CIFAR10(self.data_dir, train=True, download=True)
        CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            dataset_train = CIFAR10(self.data_dir, train=True, transform=self.transform_train)
    
            no_train = int(len(dataset_train) * 0.95)
            no_val = len(dataset_train) - no_train

            self.dataset_train, self.dataset_val = random_split(dataset_train, [no_train, no_val])
            self.num_classes = len(dataset_train.classes)
            
        if stage == 'test' or stage is None:
            self.dataset_test = CIFAR10(self.data_dir, train=False, transform=self.transform_eval)
            self.num_classes = len(self.dataset_test.classes)

    def train_dataloader(self):
        '''returns training dataloader'''
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)
        
    def val_dataloader(self):
        '''returns validation dataloader'''
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)
        
    def test_dataloader(self):
        '''returns test dataloader'''
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)
    

class CIFAR100DM(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.data_dir = args.dataset_path
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.num_workers = args.no_cpu_workers
        self.transform_train = get_transform(split='train', args=args)
        self.transform_eval = get_transform(split='val', args=args)

    def prepare_data(self):
        '''called only once and on 1 GPU'''
        # download data
        CIFAR100(self.data_dir, train=True, download=True)
        CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            dataset_train = CIFAR100(self.data_dir, train=True, transform=self.transform_train)
    
            no_train = int(len(dataset_train) * 0.95)
            no_val = len(dataset_train) - no_train

            self.dataset_train, self.dataset_val = random_split(dataset_train, [no_train, no_val])
            self.num_classes = len(dataset_train.classes)
            
        if stage == 'test' or stage is None:
            self.dataset_test = CIFAR100(self.data_dir, train=False, transform=self.transform_eval)
            self.num_classes = len(self.dataset_test.classes)

    def train_dataloader(self):
        '''returns training dataloader'''
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)
        
    def val_dataloader(self):
        '''returns validation dataloader'''
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)
        
    def test_dataloader(self):
        '''returns test dataloader'''
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)
    
            
class ImageNetDM(LightningDataModule):

    def __init__(self, args):
        super().__init__()
        self.data_dir = os.path.abspath(args.dataset_path)
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.num_workers = args.no_cpu_workers
        self.transform_train = get_transform(split='train', args=args)
        self.transform_eval = get_transform(split='val', args=args)
        
    def setup(self, stage=None):
        '''called on each GPU separately - stage defines if we are at fit or test step'''
        # we set up only relevant datasets when stage is specified (automatically set by Pytorch-Lightning)
        if stage == 'fit' or stage is None:
            data_path = os.path.join(self.data_dir, 'train')
            dataset_train = ImageFolder(root=data_path, transform=self.transform_train)
            
            no_train = int(len(dataset_train) * 0.99)
            no_val = len(dataset_train) - no_train
            
            self.dataset_train, self.dataset_val = random_split(dataset_train, [no_train, no_val])
            self.num_classes = len(dataset_train.classes)
        
        if stage == 'test' or stage is None:
            data_path = os.path.join(self.data_dir, 'val')
            self.dataset_test = ImageFolder(root=data_path, transform=self.transform_eval)
            self.num_classes = len(self.dataset_test.classes)
            
    def train_dataloader(self):
        '''returns training dataloader'''
        return DataLoader(self.dataset_train, batch_size=self.batch_size, num_workers=self.num_workers)
        
    def val_dataloader(self):
        '''returns validation dataloader'''
        return DataLoader(self.dataset_val, batch_size=self.batch_size, num_workers=self.num_workers)
        
    def test_dataloader(self):
        '''returns test dataloader'''
        return DataLoader(self.dataset_test, batch_size=self.batch_size, num_workers=self.num_workers)
        