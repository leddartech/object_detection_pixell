import torch
import torch.nn as nn


class BasicLayer(nn.Module):
    """ Basic convolution layer composed of a batch norm, activation then convolution operations.
        Works in both 2D and 3D.
    
        Args:
            in_channels - the number of input feature map channels
            out_channels - the number of output feature map channels
            kernel - the kernel for convolution in 3D format (depth, height, width), or 2D format (height, width).
            padding - if true, pads the input in order to conserve its dimensions. Similar as padding='same' for tensorflow.
            activation - activation function. Default is nn.ReLU()
            dropout - whether you want this BasicLayer to have an extra layer (after the convolution) of dropout.  
                        Specify the probability rate;  if set to None (by default), there will be no layer of dropout.
    """

    def __init__(self, in_channels:int, out_channels:int='all', kernel:tuple=(3,3,3), padding:bool=True, activation=nn.ReLU(), dropout=None):
        super(BasicLayer, self).__init__()
        self.dropout = dropout
        self.out_channels = out_channels if out_channels != 'all' else in_channels

        if padding:
            padding = tuple(int((kernel[i]-1)/2) for i in range(len(kernel)))
        else:
            padding = 0

        if len(kernel) == 3:
            self.bn = nn.BatchNorm3d(in_channels)
            self.conv = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=padding, bias=False)
            self.activation = activation
            if self.dropout:
                self.drop = nn.Dropout3d(p=dropout)
        elif len(kernel) == 2:
            self.bn = nn.BatchNorm2d(in_channels)
            self.activation = activation
            self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel, padding=padding, bias=False)
            if self.dropout:
                self.drop = nn.Dropout2d(p=dropout)

    def forward(self, x):
        x = self.bn(x)
        x = self.conv(x)
        if self.activation:
            x = self.activation(x)
        if self.dropout:
            x = self.drop(x)
        return x

class DenseBlock(nn.Module):
    """ Dense block of fully convolutional layers. It combines 'n_layers' of basic_layer instances with a 
        given number of 'growth' (output channels). The input of each basic_layer 
        is formed from the outputs of all previous basic_layer.

        Args:
            in_channels - the number of input map features that enter the DenseBlock
            out_channels - the number fo out map features that will exit the DenseBlock 
            growth - the number of output features map after each basic_layer, except the last layer (=out_channels)
            n_layers - the total of layers in DenseBlock
            kernel - the size of the convolution kernel to be used in all basic_layers
            basic_layer - the type of basic layer you want as building blocks.
            dropout - None: there is no dropout layer in basic_layer after each convolution; set probability for dropout rate.  
                    ù) Note that, there is no dropout in first and last layers. (This is the rule).
        """

    def __init__(self, in_channels, out_channels='all', growth:int=12, n_layers:int=5, kernel:tuple=(7,3,3), basic_layer=BasicLayer, dropout=None):
        super(DenseBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.growth = growth
        self.n_layers = n_layers
        self.kernel = kernel
        self.basic_layer = basic_layer
        self.dropout = dropout
        self.layers = None

        self.fin_layer = False
        if self.out_channels is not 'all':
            self.fin_layer = True
            self.transi = None
        else:
            self.out_channels = self.in_channels + self.growth*self.n_layers

        self.construct_layers()

    def construct_layers(self):
        """This construct a list of BasicLayer with the right number of in_channels at each step.
            layers = [H_1, H_2, H_3, ..., H_nlayers]
             -> nn.ModuleList(layers)
        """
        layers = []

        for omega in range(self.n_layers): ## Here we don't use dropout for first and last layer
            in_chas = self.in_channels + self.growth*omega
            if omega == 0:
                layers.append(self.basic_layer(in_channels=in_chas, out_channels=self.growth, kernel=self.kernel, padding=True, activation=nn.ReLU(), dropout=None))
            elif omega == self.n_layers-1:
                layers.append(self.basic_layer(in_channels=in_chas, out_channels=self.growth, kernel=self.kernel, padding=True, activation=None, dropout=None))
            else:
                layers.append(self.basic_layer(in_channels=in_chas, out_channels=self.growth, kernel=self.kernel, padding=True, activation=nn.ReLU(), dropout=self.dropout))

        self.layers = nn.ModuleList(layers)

        if self.fin_layer:
            if len(self.kernel) == 2:
                self.transi = BasicLayer(in_channels=self.in_channels+self.growth*self.n_layers, out_channels=self.out_channels, 
                                         kernel=(1,1), padding=True, activation=None, dropout=None)
            elif len(self.kernel) == 3:
                self.transi = BasicLayer(in_channels=self.in_channels+self.growth*self.n_layers, out_channels=self.out_channels, 
                                         kernel=(1,1,1), padding=True, activation=None, dropout=None)

    def forward(self,x):

        for omega in range(self.n_layers):
            x = torch.cat((x, self.layers[omega](x)),1)

        if self.fin_layer:
            x = self.transi(x)

        return x

class Upscale(nn.Module):
    """Decoding module.  

        It upsamples the input by a factor of 'up'.

        Operations: 
            Activation -> Batch Norm -> Trans Conv

        Args:
            in_channels - number of channel features in
            out_channels - number of channel features out
            kernel - convlution kernel size
            up_factor - upscale factor along 2 axes (height, width) or 3 axes (depth, height, width)
            activation - None, or nn.ReLU() for example.
    """
    def __init__(self, in_channels:int, out_channels:int='all', kernel:tuple=(2,1,1), up_factor:tuple=(2,1,1), activation=None):
        super(Upscale, self).__init__()

        padding = tuple(int((kernel[i]-up_factor[i])/2) for i in range(len(kernel)))

        self.out_channels = out_channels if out_channels != 'all' else in_channels

        if len(kernel) == 2: #(2d)
            self.activation = activation
            self.bn = nn.BatchNorm2d(in_channels)
            self.trans = nn.ConvTranspose2d(in_channels=in_channels, out_channels=self.out_channels, 
                                            kernel_size=kernel, stride=up_factor, padding=padding)
        
        elif len(kernel) == 3: #(3d)
            self.activation = activation
            self.bn = nn.BatchNorm3d(in_channels)
            self.trans = nn.ConvTranspose3d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=kernel, 
                                            stride=up_factor, padding=padding)

    def forward(self, x):

        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        x = self.trans(x)

        return x

class Downscale(nn.Module):
    """Decoding module.  

        It downsamples the input by a factor of 'down'.

        Operations: 
            Activation -> Batch Norm -> Trans Conv

        Args:
            in_channels - number of channel features in
            out_channels - number of channel features out
            kernel - convlution kernel size
            down_factor - downscale factor along 2 axes (height, width) or 3 axes (depth, height, width)
            activation - None, or nn.ReLU() for example.
    """
    def __init__(self, in_channels:int, out_channels:int='all', kernel:tuple=(2,1,1), down_factor:tuple=(2,1,1), activation=None):
        super(Downscale, self).__init__()

        padding = tuple(int((kernel[i]-down_factor[i])/2) for i in range(len(kernel)))

        self.out_channels = out_channels if out_channels != 'all' else in_channels

        if len(kernel) == 2: #(2d)
            self.activation = activation
            self.bn = nn.BatchNorm2d(in_channels)
            self.trans = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channels, 
                                            kernel_size=kernel, stride=down_factor, padding=padding)
        
        elif len(kernel) == 3: #(3d)
            self.activation = activation
            self.bn = nn.BatchNorm3d(in_channels)
            self.trans = nn.Conv3d(in_channels=in_channels, out_channels=self.out_channels, kernel_size=kernel, 
                                            stride=down_factor, padding=padding)

    def forward(self, x):

        x = self.bn(x)
        if self.activation:
            x = self.activation(x)
        x = self.trans(x)

        return x