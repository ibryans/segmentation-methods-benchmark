from core.loader.dataset_NL import *
from core.loader.dataset_NS import *
from core.loader.dataset_NZ import *


data_folders = {
    'NL' : 'datasets/data_NL',
    'NS' : 'datasets/data_NS',    
    'NZ' : 'datasets/data_NZ',
}


def get_loader(arch):
    if 'section' in arch:
        return section_dataset
    else:
        raise NotImplementedError()
