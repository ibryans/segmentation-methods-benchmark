# Parihaka Seismic Data - https://public.3.basecamp.com/p/JyT276MM7krjYrMoLqLQ6xST

import collections
import numpy
import os
import segyio
import torch

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


def split_train_val(args, data_folder, loader_type='section', per_val=0.1):
    """
    Create inline and crossline sections for training and validation:
        "num_inlines": 590
        "num_crosslines": 782
        "num_time_depth": 1006
    """
    labels = segyio.tools.cube(os.path.join(data_folder, 'TrainingData_Labels.segy'))
    irange, crange, depth = labels.shape

    i_list = list(range(irange))
    i_list = ['i_'+str(inline) for inline in i_list]

    c_list = list(range(crange))
    c_list = ['x_'+str(crossline) for crossline in c_list]

    list_train_val = i_list + c_list

    # create train and validation splits:
    list_eval = None
    list_train, list_val = train_test_split(list_train_val, test_size=per_val, shuffle=True, random_state=0)
    if per_val >= 0.50:
        list_val, list_eval = train_test_split(list_val, test_size=0.80, shuffle=True, random_state=0)

    # write data to files:
    file_object = open(os.path.join(data_folder, 'splits', f'{loader_type}_train_val.txt'), 'w')
    file_object.write('\n'.join(list_train_val))
    file_object.close()
    file_object = open(os.path.join(data_folder, 'splits', f'{loader_type}_train.txt'), 'w')
    file_object.write('\n'.join(list_train))
    file_object.close()
    file_object = open(os.path.join(data_folder, 'splits', f'{loader_type}_val.txt'), 'w')
    file_object.write('\n'.join(list_val))
    file_object.close()
    if list_eval is not None:
        file_object = open(os.path.join(data_folder, 'splits', f'{loader_type}_eval.txt'), 'w')
        file_object.write('\n'.join(list_eval))
        file_object.close()


def split_test_eval(args, data_folder, split, loader_type='section', per_val=0.1):
    """
    Create inline and crossline sections for testing:
        "num_inlines": 590
        "num_crosslines": 782
        "num_time_depth": 1006
    """
    if (split == "eval") and (per_val >= 0.50):
        print('[WARNING] Performance assessed on "eval" split rather than official test splits.')
        labels = segyio.tools.cube(os.path.join(data_folder, 'TrainingData_Labels.segy'))
    else:
        labels = segyio.tools.cube(os.path.join(data_folder, f'TestData_Image{split[-1]}.segy'))
    
    irange, crange, depth = labels.shape

    if args.inline:
        i_list = list(range(irange))
        i_list = ['i_'+str(inline) for inline in i_list]
    else:
        i_list = []

    if args.crossline:
        c_list = list(range(crange))
        c_list = ['x_'+str(crossline) for crossline in c_list]
    else:
        c_list = []

    list_test = i_list + c_list

    # create eval or test splits
    if (split == "eval") and (per_val >= 0.50):
        _, list_val  = train_test_split(list_test, test_size=per_val, shuffle=True, random_state=0)
        _, list_test = train_test_split(list_val, test_size=0.80, shuffle=True, random_state=0)

    # write to files to disk:
    file_object = open(os.path.join(data_folder, 'splits', f'{loader_type}_{split}.txt'), 'w')
    file_object.write('\n'.join(list_test))
    file_object.close()
        


class section_dataset(torch.utils.data.Dataset):
    """
        Data loader for the section-based deconvnet
    """
    def __init__(self, split='train', channel_delta=0, data_folder='data_NZ', is_transform=True, augmentations=None):
        self.split = split
        self.c_delta = channel_delta
        self.data_folder = data_folder
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.n_classes = 6 
        self.mean = 0.67661
        self.sections = collections.defaultdict(list)

        if self.split in ['train', 'train_val', 'val', 'eval']:
            # Normal train/val mode            
            self.seismic = segyio.tools.cube(os.path.join(self.data_folder, 'TrainingData_Image.segy'))
            self.labels  = segyio.tools.cube(os.path.join(self.data_folder, 'TrainingData_Labels.segy')).astype(numpy.uint8)-1
        elif 'test1' in self.split:
            label_path = os.path.join(self.data_folder, 'TestData_Labels1.segy')
            self.seismic = segyio.tools.cube(os.path.join(self.data_folder, 'TestData_Image1.segy'))
            self.labels  = segyio.tools.cube(label_path).astype(int)-1 if os.path.exists(label_path) else None
        elif 'test2' in self.split:
            label_path = os.path.join(self.data_folder, 'TestData_Labels2.segy')
            self.seismic = segyio.tools.cube(os.path.join(self.data_folder, 'TestData_Image2.segy'))
            self.labels  = segyio.tools.cube(label_path).astype(int)-1 if os.path.exists(label_path) else None
        else:
            raise ValueError('Unknown split: train, val, train_val, test1, test2')

        if 'test' not in self.split:
            # We are in train/val mode. Most likely the test splits are not saved yet, so don't attempt to load them.  
            for split in ['train', 'train_val', 'val', 'eval']:
                # reading the file names for 'train', 'val', 'trainval'""
                path = os.path.join(self.data_folder, 'splits', f'section_{split}.txt')
                if os.path.isfile(path):
                    file_list = tuple(open(path, 'r'))
                    file_list = [id_.rstrip() for id_ in file_list]
                    self.sections[split] = file_list
        elif 'test' in split:
            # We are in test mode. Only read the given split. The other one might not be available. 
            path = os.path.join(self.data_folder, 'splits', f'section_{split}.txt')
            file_list = tuple(open(path,'r'))
            file_list = [id_.rstrip() for id_ in file_list]
            self.sections[split] = file_list
        else:
            raise ValueError('Unknown split: train, val, train_val, test1, test2')

    def __len__(self):
        return len(self.sections[self.split])

    def __getitem__(self, index):
        """
        Dataset shape: (590, 782, 1006)
            "num_inlines": 590
            "num_crosslines": 782
            "num_time_depth": 1006
        """
        section_name = self.sections[self.split][index]
        direction, number = section_name.split(sep='_')
        slice_number = int(number)

        if direction == 'i':
            try:
                lbl = self.labels[slice_number,:,:].transpose((1,0)) if self.labels is not None else numpy.zeros(self.seismic[slice_number,:,:].transpose((1,0)).shape)
                if self.c_delta == 0:
                    img = self.seismic[slice_number,:,:].transpose((1,0))
                elif self.c_delta > 0:
                    img = self.seismic[max(0,slice_number-self.c_delta):min(self.seismic.shape[0],slice_number+self.c_delta+1),:,:]
                    img = numpy.stack([img[0,:,:], img[img.shape[0]//2,:,:], img[-1,:,:]]).transpose((0,2,1))
                else:
                    raise RuntimeError(f'INLINE - No implementation for self.c_delta={self.c_delta}')
            except:
                raise RuntimeError(f'INLINE - Batch {index}: \t section [{section_name}]={direction}_{slice_number} \t {self.seismic[slice_number,:,:].shape} {self.seismic[slice_number-self.c_delta:slice_number+self.c_delta+1,:,:].shape}')
        elif direction == 'x':  
            try:
                lbl = self.labels[:,slice_number,:].transpose((1,0)) if self.labels is not None else numpy.zeros(self.seismic[:,slice_number,:].transpose((1,0)).shape)
                if self.c_delta == 0:
                    img = self.seismic[:,slice_number,:].transpose((1,0))
                elif self.c_delta > 0:
                    img = self.seismic[:,max(0,slice_number-self.c_delta):min(self.seismic.shape[1],slice_number+self.c_delta+1),:]
                    img = numpy.stack([img[:,0,:], img[:,img.shape[1]//2,:], img[:,-1,:]]).transpose((0,2,1))
                else:
                    raise RuntimeError(f'CROSSLINE - No implementation for self.c_delta={self.c_delta}')
            except:
                raise RuntimeError(f'CROSSLINE - Batch {index}: \t section [{section_name}]={direction}_{slice_number} \t {self.seismic[:,slice_number,:].shape} {self.seismic[:,slice_number-self.c_delta:slice_number+self.c_delta+1,:].shape}')
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
        return ['basement', 'mudstone_a', 'mass_deposit', 'mudstone_b', 'slope_valley', 'canyon']
    
    def get_class_weights(self):
        return [0.84979262, 0.57790153, 0.95866329, 0.71236326, 0.99004844, 0.91123086]

    def get_prediction_horizons(self):
        return [0, 99, 199, 299, 399, 499]

    def get_seismic_colors(self):
        return numpy.asarray([ [69,117,180], [145,191,219], [224,243,248], [254,224,144], [252,141,89], [215,48,39]])

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
