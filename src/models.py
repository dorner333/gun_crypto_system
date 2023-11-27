import torch
import torch.nn.functional as F
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()
        
        # Вход [1, 28, 28]
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        # Вход [14, 14]
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        # Вход [7, 7]
        self.conv3 = nn.Conv2d(32, 49, kernel_size=3, stride=2, padding=1)
        # Вход [4, 4]
        self.fc1 = nn.Linear(784 + 784, 784)
        self.fc2 = nn.Linear(784, 784)
        
        # PReLU activation
        self.act = nn.ReLU()
        
    def forward(self, x, k):
        # x = torch.cat([x, k], dim=1)
        # Проход через сверточные слои
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        
        # Примешиваем ключ
        # x = torch.cat([x.view(x.shape[0], 1, 784), k], dim=1)
        print(x.shape, k.shape)
        x = torch.cat([x.flatten(start_dim=1), k], dim=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        
        return x
    
class DecoderBOB(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784 + 784, 784)
        self.fc2 = nn.Linear(784, 784)

        self.deconv1 = nn.ConvTranspose2d(49, 32, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 1,  kernel_size=3, stride=2, padding=1)

        self.act = nn.ReLU()

    def forward(self, x, k):
        x = torch.cat([x, k], dim=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))

        x = x.view(x.shape[0], 49, 4, 4)
        x = self.act(self.deconv1(x))
        x = self.act(self.deconv2(x))
        x = F.sigmoid(self.deconv3(x))

        return x

class DecoderEVA(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 784)
        self.fc2 = nn.Linear(784, 784)

        self.deconv1 = nn.ConvTranspose2d(49, 32, kernel_size=3, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(16, 1,  kernel_size=3, stride=2, padding=1)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))

        x = x.view(x.shape[0], 49, 4, 4)
        x = self.act(self.deconv1(x))
        x = self.act(self.deconv2(x))
        x = F.sigmoid(self.deconv3(x))

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