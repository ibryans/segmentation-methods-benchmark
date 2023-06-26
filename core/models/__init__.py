import torchvision.models as models

from core.models.deconvnet import DeconvNet, DeconvNet_skip
from core.models.segnet import SegNet
from core.models.unet import UNet
from core.models.danet.danet3 import DaNet3
from core.models.stronger_baseline.sps_cbam_seg_hrnet_ocr import HighResolutionNet, get_sps_cbam_seg_model

from core.models.filters import Canny, Gabor, Sobel


architectures = {
    'hrnet': 'HighResolutionNet',
    'danet3': 'DaNet3',
    'deconvnet': 'DeconvNet',
    'segnet': 'SegNet',
    'unet': 'UNet',
}


def get_model(name, pretrained, n_channels, n_classes):
    
    if name == 'hrnet':
        model = get_sps_cbam_seg_model()
    else:
        model = _get_model_instance(name)

    if name in ['deconvnet','patch_deconvnet']:
        model = model(n_channels=n_channels, n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=pretrained)
        model.init_vgg16_params(vgg16)
    else:
        model = model(n_classes=n_classes)

    return model

def _get_model_instance(name):
    try:
        return {
            'deconvnet': section_deconvnet,
            'deconvnet_skip': section_deconvnet_skip,
            'danet3': DaNet3
        }[name]
    except:
        print(f'Model {name} not available')
