# Based on the following implementations:
#   https://github.com/chaddy1004/sobel-operator-pytorch
#   https://github.com/zhaoyuzhi/PyTorch-Sobel
#   https://github.com/DCurro/CannyEdgePytorch
#   https://github.com/iKintosh/GaborNet


import math
import torch

from typing import Any

__all__ = ['Canny', 'Gabor', 'Sobel']


class Canny(torch.nn.Module):
    def __init__(self, in_channels=1, filter='vertical'):
        super().__init__()
        
        # Create filters
        G_00 = torch.tensor([   [ 0.0, 0.0, 0.0],
                                [ 0.0, 1.0,-1.0],
                                [ 0.0, 0.0, 0.0]])
        G_45 = torch.tensor([   [ 0.0, 0.0, 0.0],
                                [ 0.0, 1.0, 0.0],
                                [ 0.0, 0.0,-1.0]])
        G_90 = torch.tensor([   [ 0.0, 0.0, 0.0],
                                [ 0.0, 1.0, 0.0],
                                [ 0.0,-1.0, 0.0]])
        G_135 = torch.tensor([  [ 0.0, 0.0, 0.0],
                                [ 0.0, 1.0, 0.0],
                                [-1.0, 0.0, 0.0]])
        G_180 = torch.tensor([  [ 0.0, 0.0, 0.0],
                                [-1.0, 1.0, 0.0],
                                [ 0.0, 0.0, 0.0]])
        G_225 = torch.tensor([  [-1.0, 0.0, 0.0],
                                [ 0.0, 1.0, 0.0],
                                [ 0.0, 0.0, 0.0]])
        G_270 = torch.tensor([  [ 0.0,-1.0, 0.0],
                                [ 0.0, 1.0, 0.0],
                                [ 0.0, 0.0, 0.0]])
        G_315 = torch.tensor([  [ 0.0, 0.0,-1.0],
                                [ 0.0, 1.0, 0.0],
                                [ 0.0, 0.0, 0.0]])
        
        # Choose filters to compose convolution
        if filter == 'all':
            G = torch.cat([G_00.unsqueeze(0), G_45.unsqueeze(0), G_90.unsqueeze(0), G_135.unsqueeze(0), G_180.unsqueeze(0), G_225.unsqueeze(0), G_270.unsqueeze(0), G_315.unsqueeze(0)], 0)
            out_channels = 8
        elif filter == 'horizontal':
            G = torch.cat([G_00.unsqueeze(0), G_45.unsqueeze(0), G_135.unsqueeze(0), G_180.unsqueeze(0), G_225.unsqueeze(0), G_315.unsqueeze(0)], 0)
            out_channels = 6
        elif filter == 'vertical':
            G = torch.cat([G_45.unsqueeze(0), G_90.unsqueeze(0), G_135.unsqueeze(0), G_225.unsqueeze(0), G_270.unsqueeze(0), G_315.unsqueeze(0)], 0)
            out_channels = 6
        else: 
            raise RuntimeError('Invalid filter type. Choose among all, horizontal, vertical')
        G = G.unsqueeze(1)

        # Incorporate filters into network
        self.filter = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.filter.weight = torch.nn.Parameter(G, requires_grad=False)

    def forward(self, img, concat=False):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        if concat:
            x = torch.cat((img, x.detach()), 1)
        return x


class Sobel(torch.nn.Module):
    def __init__(self, in_channels=1, filter='vertical'):
        super().__init__()
        
        # Create filters
        Gx  = torch.tensor([    [ 1.0, 0.0,-1.0], 
                                [ 2.0, 0.0,-2.0], 
                                [ 1.0, 0.0,-1.0]])
        Gy  = torch.tensor([    [ 1.0, 2.0, 1.0], 
                                [ 0.0, 0.0, 0.0], 
                                [-1.0,-2.0,-1.0]])
        Gd1 = torch.tensor([    [ 0.0, 1.0, 2.0], 
                                [-1.0, 0.0, 1.0], 
                                [-2.0,-1.0, 0.0]])
        Gd2 = torch.tensor([    [-2.0,-1.0, 0.0], 
                                [-1.0, 0.0, 1.0], 
                                [ 0.0, 1.0, 2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0), Gd1.unsqueeze(0), Gd2.unsqueeze(0)], 0)
        

        # Choose filters to compose convolution
        if filter == 'all':
            G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0), Gd1.unsqueeze(0), Gd2.unsqueeze(0)], 0)
            out_channels = 4
        elif filter == 'horizontal':
            G = torch.cat([Gx.unsqueeze(0), Gd1.unsqueeze(0), Gd2.unsqueeze(0)], 0)
            out_channels = 3
        elif filter == 'vertical':
            G = torch.cat([Gy.unsqueeze(0), Gd1.unsqueeze(0), Gd2.unsqueeze(0)], 0)
            out_channels = 3
        else: 
            raise RuntimeError('Invalid filter type. Choose among all, horizontal, vertical')
        G = G.unsqueeze(1)

        # Incorporate filters into network
        self.filter = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.filter.weight = torch.nn.Parameter(G, requires_grad=False)

    def forward(self, img, concat=False):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        if concat:
            x = torch.cat((img, x.detach()), 1)
        return x


class GaborConv2d(torch.nn.Module):
    """
    freq, theta, sigma are set up according to S. Meshgini, A. Aghagolzadeh and H. Seyedarabi, "Face recognition using Gabor filter bank, kernel principal component analysis and support vector machine"
    """
    def __init__(
        self,
        in_channels,
        out_channels=32,
        kernel_size=(15,15),
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        padding_mode="zeros",
    ):
        super().__init__()

        self.is_calculated = False

        self.conv_layer = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.kernel_size = self.conv_layer.kernel_size

        # Small addition to avoid division by zero
        self.delta = 1e-3

        # Setting freq, theta, sigma
        self.freq = torch.nn.Parameter(
            (math.pi / 2)
            * math.sqrt(2)
            ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor),
            requires_grad=True,
        )
        self.theta = torch.nn.Parameter(
            (math.pi / 8)
            * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor),
            requires_grad=True,
        )
        self.sigma = torch.nn.Parameter(math.pi / self.freq, requires_grad=True)
        self.psi = torch.nn.Parameter(
            math.pi * torch.rand(out_channels, in_channels), requires_grad=True
        )

        self.x0 = torch.nn.Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0], requires_grad=False
        )
        self.y0 = torch.nn.Parameter(
            torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0], requires_grad=False
        )

        self.y, self.x = torch.meshgrid(
            [
                torch.linspace(-self.x0 + 1, self.x0 + 0, self.kernel_size[0]),
                torch.linspace(-self.y0 + 1, self.y0 + 0, self.kernel_size[1]),
            ],
            indexing='ij'
        )
        self.y = torch.nn.Parameter(self.y)
        self.x = torch.nn.Parameter(self.x)

        self.weight = torch.nn.Parameter(
            torch.empty(self.conv_layer.weight.shape, requires_grad=True),
            requires_grad=True,
        )

        self.register_parameter("freq", self.freq)
        self.register_parameter("theta", self.theta)
        self.register_parameter("sigma", self.sigma)
        self.register_parameter("psi", self.psi)
        self.register_parameter("x_shape", self.x0)
        self.register_parameter("y_shape", self.y0)
        self.register_parameter("y_grid", self.y)
        self.register_parameter("x_grid", self.x)
        self.register_parameter("weight", self.weight)

    def forward(self, input_tensor):
        if self.training:
            self.calculate_weights()
            self.is_calculated = False
        if not self.training:
            if not self.is_calculated:
                self.calculate_weights()
                self.is_calculated = True
        return self.conv_layer(input_tensor)

    def calculate_weights(self):
        for i in range(self.conv_layer.out_channels):
            for j in range(self.conv_layer.in_channels):
                sigma = self.sigma[i, j].expand_as(self.y)
                freq = self.freq[i, j].expand_as(self.y)
                theta = self.theta[i, j].expand_as(self.y)
                psi = self.psi[i, j].expand_as(self.y)

                rotx = self.x * torch.cos(theta) + self.y * torch.sin(theta)
                roty = -self.x * torch.sin(theta) + self.y * torch.cos(theta)

                g = torch.exp(
                    -0.5 * ((rotx ** 2 + roty ** 2) / (sigma + self.delta) ** 2)
                )
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * math.pi * sigma ** 2)
                self.conv_layer.weight.data[i, j] = g

    def _forward_unimplemented(self, *inputs: Any):
        """
        code checkers makes implement this method,
        looks like error in PyTorch
        """
        raise NotImplementedError


class Gabor(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=64):
        super().__init__()
        self.filter = GaborConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(9,9), stride=1, padding=4, bias=False)

    def forward(self, img, concat=False):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        if concat:
            x = torch.cat((img, x.detach()), 1)
        return x


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from PIL import Image
    from torchvision import transforms

    image = Image.open("woman.jpg")
    transform = transforms.Compose([transforms.PILToTensor()])
    tensor_img = transform(image).unsqueeze(0).float()
    print('tensor_img', tensor_img.shape)

    tensor_cat = torch.cat((tensor_img, tensor_img), 1)
    print('tensor_cat', tensor_cat.shape)


    orientation='vertical'

    canny_filter = Canny(in_channels=1, filter=orientation)
    tensor_filter = canny_filter(tensor_img)
    plt.figure(dpi=90, figsize=(10,10))
    plt.imshow(tensor_filter.squeeze().detach().numpy(), cmap='gray')
    plt.savefig(f'canny_{orientation}.jpg')

    sobel_filter = Sobel(in_channels=1, filter=orientation)
    tensor_filter = sobel_filter(tensor_img)
    plt.figure(dpi=90, figsize=(10,10))
    plt.imshow(tensor_filter.squeeze().detach().numpy(), cmap='gray')
    plt.savefig(f'sobel_{orientation}.jpg')

    gabor_filter = Gabor(in_channels=1, out_channels=16)
    tensor_filter = gabor_filter(tensor_img)
    plt.figure(dpi=90, figsize=(10,10))
    plt.imshow(tensor_filter.squeeze().detach().numpy(), cmap='gray')
    plt.savefig('gabor.jpg')
