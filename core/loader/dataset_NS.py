# Penobscot Interpretation Dataset - https://zenodo.org/record/3924682#.Y8aZz3bMIbU

import collections
import h5py
import numpy
import os
import torch

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def split_train_val(args, data_folder, loader_type='section', per_val=0.1, train_range=[358,1500,289]):
    """
    Create inline and crossline sections for training and validation:
        "num_inlines": 601
        "num_time_depth": 1501
        "num_crosslines": 481
    """
    dataset = h5py.File(os.path.join(data_folder, 'dataset.h5'))
    labels = dataset['label']
    irange, depth, crange = labels.shape

    i_list = list(range(irange)) if train_range is None else list(range(train_range[0]))
    i_list = ['i_'+str(inline) for inline in i_list]

    c_list = list(range(crange)) if train_range is None else list(range(train_range[2]))
    c_list = ['x_'+str(crossline) for crossline in c_list]

    list_train_val = i_list + c_list

    # create train and test splits:
    list_train, list_val = train_test_split(list_train_val, test_size=per_val, shuffle=True)

    # write to files to disk:
    file_object = open(os.path.join(data_folder, 'splits', f'{loader_type}_train_val.txt'), 'w')
    file_object.write('\n'.join(list_train_val))
    file_object.close()
    file_object = open(os.path.join(data_folder, 'splits', f'{loader_type}_train.txt'), 'w')
    file_object.write('\n'.join(list_train))
    file_object.close()
    file_object = open(os.path.join(data_folder, 'splits', f'{loader_type}_val.txt'), 'w')
    file_object.write('\n'.join(list_val))
    file_object.close()


def split_test_eval(args, data_folder, split='test1', loader_type='section', per_val=None, test_range=[358,1500,289]):
    """
    Create inline and crossline sections for testing:
        "num_inlines": 601
        "num_time_depth": 1501
        "num_crosslines": 481
    """
    dataset = h5py.File(os.path.join(data_folder, 'dataset.h5'))
    labels = dataset['label']
    irange, depth, crange = labels.shape

    if args.inline:
        i_list = list(range(test_range[0], irange))
        i_list = ['i_'+str(inline) for inline in i_list]
    else:
        i_list = []

    if args.crossline:
        c_list = list(range(test_range[2], crange))
        c_list = ['x_'+str(crossline) for crossline in c_list]
    else:
        c_list = []

    # create test split
    list_test = i_list + c_list

    # write to files to disk:
    file_object = open(os.path.join(data_folder, 'splits', f'{loader_type}_{split}.txt'), 'w')
    file_object.write('\n'.join(list_test))
    file_object.close()


class section_dataset(torch.utils.data.Dataset):
    """
        Data loader for the section-based deconvnet
    """
    def __init__(self, split='train', channel_delta=0, data_folder='data_NS', is_transform=True, augmentations=None):
        self.split = split
        self.c_delta = channel_delta
        self.data_folder = data_folder
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 8
        self.mean = -1.2681936
        self.sections = collections.defaultdict(list)

        if self.split in ['train', 'val', 'train_val', 'test1', 'test2', 'test']:
            # Normal train/val mode
            dataset = h5py.File(os.path.join(data_folder, 'dataset.h5'))
            self.seismic = numpy.squeeze(dataset['features'])
            self.labels  = dataset['label']
        else:
            raise ValueError(f'Unknown split {split} not among [train, val, train_val, test]')

        if self.split in ['train', 'val', 'train_val']:
            # We are in train/val mode. Most likely the test splits are not saved yet, so don't attempt to load them.  
            for split in ['train', 'val', 'train_val']:
                # reading the file names for 'train', 'val', 'trainval'""
                path = os.path.join(self.data_folder, 'splits', f'section_{self.split}.txt')
                file_list = tuple(open(path, 'r'))
                file_list = [id_.rstrip() for id_ in file_list]
                self.sections[split] = file_list
        elif self.split in ['test1', 'test2', 'test']:
            # We are in test mode. Only read the given split. The other one might not be available. 
            path = os.path.join(self.data_folder, 'splits', f'section_{self.split}.txt')
            file_list = tuple(open(path,'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.sections[split] = file_list
        else:
            raise ValueError('Unknown split: train, val, train_val, test')

    def __len__(self):
        return len(self.sections[self.split])

    def __getitem__(self, index):
        """
        Dataset shape: (601, 1501, 481)
            "num_inlines": 601,
            "num_time_depth": 1501,
            "num_crosslines": 481,
        """
        section_name = self.sections[self.split][index]
        direction, number = section_name.split(sep='_')
        slice_number = int(number)

        if direction == 'i':
            try:
                lbl = self.labels[slice_number,:,:]
                if self.c_delta == 0:
                    img = self.seismic[slice_number,:,:] # dimension already set to [H x W]
                elif self.c_delta > 0:
                    img = self.seismic[max(0,slice_number-self.c_delta):min(self.seismic.shape[0],slice_number+self.c_delta+1),:,:]
                    img = numpy.stack([img[0,:,:], img[img.shape[0]//2,:,:], img[-1,:,:]]) # dimension already set to [C x H x W]
                else:
                    raise RuntimeError(f'INLINE - No implementation for self.c_delta={self.c_delta}')
            except:
                raise RuntimeError(f'INLINE - Batch {index}: \t section [{section_name}]={direction}_{slice_number} \t {self.seismic[slice_number,:,:].shape} {self.seismic[slice_number-self.c_delta:slice_number+self.c_delta+1,:,:].shape}')
        elif direction == 'x':  
            try:
                lbl = self.labels[:,:,slice_number].transpose((1,0))
                if self.c_delta == 0:
                    img = self.seismic[:,:,slice_number].transpose((1,0)) # tranpose and set dimension to [H x W]
                elif self.c_delta > 0:
                    img = self.seismic[:,:,max(0,slice_number-self.c_delta):min(self.seismic.shape[2],slice_number+self.c_delta+1)]
                    img = numpy.stack([img[:,:,0], img[:,:,img.shape[2]//2], img[:,:,-1]]).transpose((0,2,1)) # tranpose and set dimension to [C x H x W]
                else:
                    raise RuntimeError(f'CROSSLINE - No implementation for self.c_delta={self.c_delta}')
            except:
                raise RuntimeError(f'CROSSLINE - Batch {index}: \t section [{section_name}]={direction}_{slice_number} \t {self.seismic[:,:,slice_number].shape} {self.seismic[:,slice_number-self.c_delta:slice_number+self.c_delta+1,:].shape}')
        else: 
            raise RuntimeError(f'Seismic direction does not correspond to INLINE (I) or CROSSLINE (X)')
        
        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)
            
        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        return img, lbl

    def transform(self, img, lbl):
        img -= self.mean

        # to be in the BxCxHxW that PyTorch uses: 
        # if len(img.shape) == 2:
            # img, lbl = img.T, lbl.T

        # to be in the BxCxHxW that PyTorch uses: 
        if len(img.shape) == 2:
            img = numpy.expand_dims(img,0)
        if len(lbl.shape) == 2:
            lbl = numpy.expand_dims(lbl,0)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
                
        return img, lbl
    
    def get_class_names(self):
        return ['h0', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'h7']
    
    def get_class_weights(self):
        return [0.95916113, 0.94736984, 0.9350648, 0.98740968, 0.9802398, 0.84783153, 0.9148715, 0.42805171]

    def get_prediction_horizons(self):
        return [0, 29, 129, 229, 329, 429]

    def get_seismic_colors(self):
        return numpy.asarray([ [69,117,180], [145,191,219], [224,243,248], [254,224,144], [252,141,89], [215,48,39], [135,35,215], [215,0,255]])

    def decode_segmap(self, label_mask, plot=False, save_name=None):
        """Decode segmentation class labels into a color image
        Args:
            label_mask (numpy.ndarray): an (M,N) array of integer values denoting
              the class label at each spatial location.
            plot (bool, optional): whether to show the resulting color image
              in a figure.
        Returns:
            (numpy.ndarray, optional): the resulting decoded color image.
        """
        label_colours = self.get_seismic_colors()
        r = label_mask.copy()
        g = label_mask.copy()
        b = label_mask.copy()
        for ll in range(0, self.n_classes):
            r[label_mask == ll] = label_colours[ll, 0]
            g[label_mask == ll] = label_colours[ll, 1]
            b[label_mask == ll] = label_colours[ll, 2]
        rgb = numpy.zeros((label_mask.shape[0], label_mask.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        plt.tight_layout()
        if save_name is not None: 
            plt.imshow(rgb)
            plt.savefig(save_name)
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb
