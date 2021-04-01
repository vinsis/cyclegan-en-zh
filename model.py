'''
Code taken directly from:
https://github.com/aitorzip/PyTorch-CycleGAN/blob/master/models.py

with minor modifications made
'''

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        model = [  nn.ReflectionPad2d(3),
                   nn.Conv2d(input_nc, 64, 7),
                   nn.InstanceNorm2d(64),
                   nn.ReLU(inplace=True)  ]
        
        # downsample
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)  ]
            in_features = out_features
            out_features = in_features * 2

        # residual blocks
        for _ in range(n_residual_blocks):
            model += [ ResidualBlock(in_features) ]

        # upsample
        out_features = in_features // 2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)  ]
            in_features = out_features
            out_features = in_features // 2

        # final layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(in_features, output_nc, 7),
                    nn.Tanh()  ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, input_nc, N=64):
        super(Discriminator, self).__init__()

        model = [  nn.Conv2d(input_nc, N, 4, stride=2, padding=1),
                   nn.LeakyReLU(0.2, inplace=True),

                   nn.Conv2d(N, N*2, 4, stride=2, padding=1),
                   nn.InstanceNorm2d(N*2),
                   nn.LeakyReLU(0.2, inplace=True),

                   nn.Conv2d(N*2, N*4, 4, stride=2, padding=1),
                   nn.InstanceNorm2d(N*4),
                   nn.LeakyReLU(0.2, inplace=True),
                   
                   nn.Conv2d(N*4, N*8, 4, padding=1),
                   nn.InstanceNorm2d(N*8),
                   nn.LeakyReLU(0.2, inplace=True)  ]

        # classification layer
        model += [ nn.Conv2d(512, 1, 4, padding=1) ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return F.avg_pool2d(x, x.size()[2:]).view(x.size(0), -1)


if __name__ == '__main__':
    import torch
    x = torch.randn(4,3,100,100)
    print('x.size():', x.size())

    print('Testing generator')
    with torch.no_grad():
        y = Generator(3,3)(x)
    print('y.size():', y.size())

    print('Testing discriminator')
    with torch.no_grad():
        z = Discriminator(3)(y)
        print('z.size():', z.size())