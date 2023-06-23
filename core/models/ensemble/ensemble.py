"""
@autor Bryan Oliveira
2023

Implementation of Ensemble learning method for segnet and unet 
(Seismic Facies Analysis Based on Deep Learning)
"""

# importing voting classifier
from sklearn.ensemble import VotingClassifier
from core.models.ensemble import SegNet, UNet


def train(architecture, dataset):
    ...


if __name__ == '__main__':
    
    # Treina os 4 modelos
    model_1 = train(SegNet(), 'data1-subset1')
    model_2 = train(SegNet(), 'data1-subset2')
    model_3 = train(UNet(), 'data2-subset-1')
    model_4 = train(UNet(), 'data2-subset-2)
 
    # Cria um modelo final usando voting classifier
    final_model = VotingClassifier(
        estimators=[('segnet-1', model_1), 
                    ('segnet-2', model_2), 
                    ('unet-1', model_3), 
                    ('unet-2', model_4)], 
        voting='hard')
                    
    