# SegNet's encoder-decoder architecture based off VGG16
# https://ieeexplore.ieee.org/abstract/document/7803544


import torch

class SegNet(torch.nn.Module):
    def __init__(self, n_channels=3, n_classes=19):
        super(SegNet, self).__init__()

        batchNorm_momentum = 0.1

        self.conv11 = torch.nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        self.bn11 = torch.nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv12 = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12 = torch.nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv21 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn21 = torch.nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv22 = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22 = torch.nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv31 = torch.nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn31 = torch.nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32 = torch.nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv33 = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33 = torch.nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv41 = torch.nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn41 = torch.nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42 = torch.nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv43 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43 = torch.nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv51 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51 = torch.nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52 = torch.nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv53 = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53 = torch.nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv53d = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn53d = torch.nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv52d = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn52d = torch.nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv51d = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn51d = torch.nn.BatchNorm2d(512, momentum=batchNorm_momentum)

        self.conv43d = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn43d = torch.nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv42d = torch.nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn42d = torch.nn.BatchNorm2d(512, momentum=batchNorm_momentum)
        self.conv41d = torch.nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn41d = torch.nn.BatchNorm2d(256, momentum=batchNorm_momentum)

        self.conv33d = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn33d = torch.nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv32d = torch.nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn32d = torch.nn.BatchNorm2d(256, momentum=batchNorm_momentum)
        self.conv31d = torch.nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn31d = torch.nn.BatchNorm2d(128, momentum=batchNorm_momentum)

        self.conv22d = torch.nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn22d = torch.nn.BatchNorm2d(128, momentum=batchNorm_momentum)
        self.conv21d = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn21d = torch.nn.BatchNorm2d(64, momentum=batchNorm_momentum)

        self.conv12d = torch.nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn12d = torch.nn.BatchNorm2d(64, momentum=batchNorm_momentum)
        self.conv11d = torch.nn.Conv2d(64, n_classes, kernel_size=3, padding=1)

    def forward(self, x):
        # Stage 1
        x11 = torch.nn.functional.relu(self.bn11(self.conv11(x)))
        x12 = torch.nn.functional.relu(self.bn12(self.conv12(x11)))
        x1_size = x12.size()
        x1p, id1 = torch.nn.functional.max_pool2d(x12, kernel_size=2, stride=2, return_indices=True)

        # Stage 2
        x21 = torch.nn.functional.relu(self.bn21(self.conv21(x1p)))
        x22 = torch.nn.functional.relu(self.bn22(self.conv22(x21)))
        x2_size = x22.size()
        x2p, id2 = torch.nn.functional.max_pool2d(x22, kernel_size=2, stride=2, return_indices=True)

        # Stage 3
        x31 = torch.nn.functional.relu(self.bn31(self.conv31(x2p)))
        x32 = torch.nn.functional.relu(self.bn32(self.conv32(x31)))
        x33 = torch.nn.functional.relu(self.bn33(self.conv33(x32)))
        x3_size = x33.size()
        x3p, id3 = torch.nn.functional.max_pool2d(x33, kernel_size=2, stride=2, return_indices=True)

        # Stage 4
        x41 = torch.nn.functional.relu(self.bn41(self.conv41(x3p)))
        x42 = torch.nn.functional.relu(self.bn42(self.conv42(x41)))
        x43 = torch.nn.functional.relu(self.bn43(self.conv43(x42)))
        x4_size = x43.size()
        x4p, id4 = torch.nn.functional.max_pool2d(x43, kernel_size=2, stride=2, return_indices=True)

        # Stage 5
        x51 = torch.nn.functional.relu(self.bn51(self.conv51(x4p)))
        x52 = torch.nn.functional.relu(self.bn52(self.conv52(x51)))
        x53 = torch.nn.functional.relu(self.bn53(self.conv53(x52)))
        x5_size = x53.size()
        x5p, id5 = torch.nn.functional.max_pool2d(x53, kernel_size=2, stride=2, return_indices=True)

        # Stage 5d
        x5d = torch.nn.functional.max_unpool2d(x5p, id5, kernel_size=2, stride=2, output_size=x5_size)
        x53d = torch.nn.functional.relu(self.bn53d(self.conv53d(x5d)))
        x52d = torch.nn.functional.relu(self.bn52d(self.conv52d(x53d)))
        x51d = torch.nn.functional.relu(self.bn51d(self.conv51d(x52d)))

        # Stage 4d
        x4d = torch.nn.functional.max_unpool2d(x51d, id4, kernel_size=2, stride=2, output_size=x4_size)
        x43d = torch.nn.functional.relu(self.bn43d(self.conv43d(x4d)))
        x42d = torch.nn.functional.relu(self.bn42d(self.conv42d(x43d)))
        x41d = torch.nn.functional.relu(self.bn41d(self.conv41d(x42d)))

        # Stage 3d
        x3d = torch.nn.functional.max_unpool2d(x41d, id3, kernel_size=2, stride=2, output_size=x3_size)
        x33d = torch.nn.functional.relu(self.bn33d(self.conv33d(x3d)))
        x32d = torch.nn.functional.relu(self.bn32d(self.conv32d(x33d)))
        x31d = torch.nn.functional.relu(self.bn31d(self.conv31d(x32d)))

        # Stage 2d
        x2d = torch.nn.functional.max_unpool2d(x31d, id2, kernel_size=2, stride=2, output_size=x2_size)
        x22d = torch.nn.functional.relu(self.bn22d(self.conv22d(x2d)))
        x21d = torch.nn.functional.relu(self.bn21d(self.conv21d(x22d)))

        # Stage 1d
        x1d = torch.nn.functional.max_unpool2d(x21d, id1, kernel_size=2, stride=2, output_size=x1_size)
        x12d = torch.nn.functional.relu(self.bn12d(self.conv12d(x1d)))
        x11d = self.conv11d(x12d)

        return x11d