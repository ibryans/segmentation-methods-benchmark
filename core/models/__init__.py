import torchvision.models as models

from core.models.deconvnet import DeconvNet
from core.models.segnet import SegNet
from core.models.unet import UNet
from core.models.danet.danet3 import DaNet3
from core.models.stronger_baseline.sps_cbam_seg_hrnet_ocr import get_sps_cbam_seg_model

architectures = {
    'hrnet': get_sps_cbam_seg_model,
    'danet3': DaNet3,
    'deconvnet': DeconvNet,
    'segnet': SegNet,
    'unet': UNet,
}

def get_model(name, pretrained, n_channels, n_classes):

    if name in ['deconvnet','patch_deconvnet']:
        model = model(n_channels=n_channels, n_classes=n_classes)
        vgg16 = models.vgg16(pretrained=pretrained)
        model.init_vgg16_params(vgg16)
    else:
        model = architectures[name](n_classes=n_classes)

    return model
