import torch
import torch.nn.functional as F
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, key_size: int):
        super(Encoder, self).__init__()
        
        # Вход [1, 28, 28]
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        # Вход [14, 14]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        # Вход [7, 7]
        self.conv3 = nn.Conv2d(32, 49, kernel_size=3, stride=2, padding=1)
        # Вход [4, 4]
        
        self.key_expander1 = nn.Sequential(nn.Linear(key_size, 16 * 2), nn.SiLU())
        self.key_expander2 = nn.Sequential(nn.Linear(key_size, 32 * 2), nn.SiLU())
        self.key_expander3 = nn.Sequential(nn.Linear(key_size, 49 * 2), nn.SiLU())

        # PReLU activation
        self.act = nn.PReLU()
        
    def forward(self, x: torch.Tensor, k: torch.Tensor):
        x = self.conv1(x)
        k_expanded = self.key_expander1(k)[:, :, None, None]
        scale, shift = k_expanded.chunk(2, dim=1)
        x = x * (1 + scale) + shift
        x = self.act(x)

        x = self.conv2(x)
        k_expanded = self.key_expander2(k)[:, :, None, None]
        scale, shift = k_expanded.chunk(2, dim=1)
        x = x * (1 + scale) + shift
        x = self.act(x)

        x = self.conv3(x)
        k_expanded = self.key_expander3(k)[:, :, None, None]
        scale, shift = k_expanded.chunk(2, dim=1)
        x = x * (1 + scale) + shift
        x = x - x.min(1, keepdim=True)[0]
        x = x / x.max(1, keepdim=True)[0]
        return x
    
class DecoderBOB(nn.Module):
    def __init__(self, key_size):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(49, 32, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 1,  kernel_size=3, stride=2, padding=1, output_padding=1)

        self.key_expander1 = nn.Sequential(nn.Linear(key_size, 49 * 2), nn.SiLU())
        self.key_expander2 = nn.Sequential(nn.Linear(key_size, 32 * 2), nn.SiLU())
        self.key_expander3 = nn.Sequential(nn.Linear(key_size, 16 * 2), nn.SiLU())

        self.act = nn.PReLU()

    def forward(self, x, k):

        k_expanded = self.key_expander1(k)[:, :, None, None]
        scale, shift = k_expanded.chunk(2, dim=1)
        x = (x + shift) * (1 + scale)
        x = self.deconv1(x)
        x = self.act(x)

        k_expanded = self.key_expander2(k)[:, :, None, None]
        scale, shift = k_expanded.chunk(2, dim=1)
        x = (x + shift) * (1 + scale)
        x = self.deconv2(x)
        x = self.act(x)

        k_expanded = self.key_expander3(k)[:, :, None, None]
        scale, shift = k_expanded.chunk(2, dim=1)
        x = (x + shift) * (1 + scale)
        x = self.deconv3(x)

        return x

class DecoderEVA(nn.Module):
    def __init__(self):
        super().__init__()

        self.deconv1 = nn.ConvTranspose2d(49, 32, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 1,  kernel_size=3, stride=2, padding=1, output_padding=1)

        self.scale1 = nn.Parameter(torch.randn(1, 49, 1, 1))
        self.scale2 = nn.Parameter(torch.randn(1, 32, 1, 1))
        self.scale3 = nn.Parameter(torch.randn(1, 16, 1, 1))

        self.shift1 = nn.Parameter(torch.randn(1, 49, 1, 1))
        self.shift2 = nn.Parameter(torch.randn(1, 32, 1, 1))
        self.shift3 = nn.Parameter(torch.randn(1, 16, 1, 1))

        self.act = nn.PReLU()

    def forward(self, x):

        x = (x + self.shift1) * (1 + self.scale1)
        x = self.deconv1(x)
        x = self.act(x)

        x = (x + self.shift2) * (1 + self.scale2)
        x = self.deconv2(x)
        x = self.act(x)

        x = (x + self.shift3) * (1 + self.scale3)
        x = self.deconv3(x)

        return x
    


class MixTransformNN(torch.nn.Module):
    def __init__(self, D_in, H):

        super(MixTransformNN, self).__init__()
        self.fc_layer = torch.nn.Linear(D_in, H)
        self.conv1 = torch.nn.Conv1d(in_channels=1,
                                    out_channels=2,
                                    kernel_size=4,
                                    stride=1,
                                    padding=2)
        self.conv2 = torch.nn.Conv1d(in_channels=2,
                                    out_channels=4,
                                    kernel_size=2,
                                    stride=2)
        self.conv3 = torch.nn.Conv1d(in_channels=4,
                                    out_channels=4,
                                    kernel_size=1,
                                    stride=1)
        self.conv4 = torch.nn.Conv1d(in_channels=4,
                                    out_channels=1,
                                    kernel_size=1,
                                    stride=1)
    # end

    def forward(self, x):

        x = x[None, :, :].transpose(0, 1)

        x = F.sigmoid(self.fc_layer(x))

        x = F.sigmoid(self.conv1(x))

        x = F.sigmoid(self.conv2(x))

        x = F.sigmoid(self.conv3(x))

        x = F.tanh(self.conv4(x))

        return torch.squeeze(x)