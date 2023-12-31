import torch
import torch.nn.functional as F
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,):
        super(Encoder, self).__init__()
        
        # Вход [1, 28, 28]
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 96, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(96, 288, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(288, 784, kernel_size=5, stride=2, padding=1)
        # Выход [784, 1, 1]
        
        self.fc1 = nn.Linear(784 + 128, 784)
        self.fc2 = nn.Linear(784, 784)
        self.fc3 = nn.Linear(784, 784)
        
        # PReLU activation
        self.prelu = nn.PReLU()
        
    def forward(self, x, k):
        # Проход через сверточные слои
        x = self.prelu(self.conv1(x))
        x = self.prelu(self.conv2(x))
        x = self.prelu(self.conv3(x))
        x = self.prelu(self.conv4(x))
        
        # Примешиваем ключ
        # x = torch.cat([x.view(x.shape[0], 1, 784), k], dim=1)
        x = torch.cat([x.view(x.shape[0], 1, 784), k], dim=-1)
        
        x = self.prelu(self.fc1(x))
        x = self.prelu(self.fc2(x))
        x = nn.Tanh()(self.fc3(x)) 
        
        return x
    
class DecoderBOB(nn.Module):
    def __init__(self):
        super(DecoderBOB, self).__init__()

        self.fc4 = nn.Linear(784 + 128, 784)
        self.fc5 = nn.Linear(784, 784)
        self.fc6 = nn.Linear(784, 784)

        self.deconv4 = nn.ConvTranspose2d(784, 288, kernel_size=5, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(288, 96, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(96, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.prelu = nn.PReLU()

    def forward(self, x, k):
        x = self.prelu(self.fc4(torch.cat([x, k], dim=-1)))
        x = self.prelu(self.fc5(x))
        x = self.prelu(self.fc6(x))

        x = x.view(x.shape[0], 784, 1, 1)

        x = self.prelu(self.deconv4(x))
        x = self.prelu(self.deconv3(x))
        x = self.prelu(self.deconv2(x))
        x = nn.Tanh()(self.deconv1(x)) 

        return x

class DecoderEVA(nn.Module):
    def __init__(self):
        super(DecoderEVA, self).__init__()

        self.fc4 = nn.Linear(784, 784)
        self.fc5 = nn.Linear(784, 784)
        self.fc6 = nn.Linear(784, 784)

        self.deconv4 = nn.ConvTranspose2d(784, 288, kernel_size=5, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(288, 96, kernel_size=3, stride=2, padding=1, output_padding=0)
        self.deconv2 = nn.ConvTranspose2d(96, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv1 = nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.prelu(self.fc4(x))
        x = self.prelu(self.fc5(x))
        x = self.prelu(self.fc6(x))

        x = x.view(x.shape[0], 784, 1, 1)

        x = self.prelu(self.deconv4(x))
        x = self.prelu(self.deconv3(x))
        x = self.prelu(self.deconv2(x))
        x = nn.Tanh()(self.deconv1(x))  

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
    # end
# end
