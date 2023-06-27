"""
Implementation of DaNet-3 
(Efficient Classification of Seismic Textures)
"""

# Pytorch
from torch import nn

from core.models.utils import conv2DBatchNormRelu


class DaNet3(nn.Module):
    def __init__(self, input_channels=1, classes=7, **kwargs):
        
        super(DaNet3, self).__init__(**kwargs)
        
        self.features = nn.Sequential(   
            # Entrada padrão = 40x40x1
            # Utilizando padding="same"
            
            # TODO: Substituir os blocos pela função da utils.py conv2DBatchNormRelu
            # -> conv2DBatchNormRelu(in_channels=input_channels, n_filters=64, k_size=(5,5), stride=2, padding='same')

            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=(5,5), stride=2, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=2, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=2, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding="same"),
            nn.ReLU(),
            nn.BatchNorm2d(),
        )
        
        # Camada totalmente conectada
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 2048),
            nn.Linear(2048, classes),
            nn.Softmax()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x