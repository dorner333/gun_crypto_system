import torch
import torch.nn as nn


class CouplingLayer(nn.Module):
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        super(CouplingLayer, self).__init__()
        self.model = model.to(device)
        self.device = device

    def forward(self, x: torch.Tensor, key: torch.Tensor, reverse=False):
        # print("kldfjkhsd")
        eps = 1e-6
        if not reverse:
            a, b = self.model(key)
            print(a.abs().min())
            out = (a+eps) * x + (b+eps)
            return out

        if reverse:
            a, b = self.model(key)
            out = (x - (b+eps)) / (a+eps)
            return out

    @property
    def coupling_parameters(self):
        return [(param.data.cpu().numpy(), param.grad.data.cpu().numpy())
                for param in self.model.parameters()]


class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(784, 784)
        self.fc2 = nn.Linear(784, 784*2)
        self.fc3 = nn.Linear(784*2, 784*4)
        self.fc4 = nn.Linear(784*4, 784*2)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.prelu(self.fc1(x))
        y = self.prelu(self.fc2(y))
        y = self.prelu(self.fc3(y))
        y = self.prelu(self.fc4(y))
        y = torch.reshape(y, (2, 28, 28))  # возможно (1, 2, 28, 28)
        return y[0], y[1]

    @property
    def fcn_parameters(self):
        return [(param.data.cpu().numpy(), param.grad.data.cpu().numpy())
                for param in self.parameters()]


class Encryptor(nn.Module):
    def __init__(self, num_blocks: int = 1) -> None:
        super(Encryptor, self).__init__()
        self.num_blocks = num_blocks
        self.couplings = nn.ModuleList([CouplingLayer(FCN(), device='cuda:0')
                                        for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor, key: torch.Tensor, reverse=False):
        if not reverse:
            for block in self.couplings:
                x = block(x, key, reverse=False)
            return x
        if reverse:
            for block in reversed(self.couplings):
                x = block(x, key, reverse=True)
            return x

    @property
    def encryptor_parameters(self):
        return {'block_{}'.format(i): block.coupling_parameters
                for i, block in enumerate(self.blocks)}


if __name__ == "__main__":
    encryptor = Encryptor(num_blocks=10)
    import torchvision.datasets
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
        ])
    mnist = torchvision.datasets.MNIST('./', train=False, transform=transform)
    img, _ = mnist.__getitem__(0)
    dummy_input = img.to('cuda:0')
    key = torch.randint(0, 2, (1, 784)).to('cuda:0', dtype=torch.float32)
    cyphroimage = encryptor(dummy_input, key)
    # print(dummy_input)
    print((encryptor(cyphroimage, key, reverse=True) - dummy_input).sum(),
          dummy_input.mean(), cyphroimage.mean())
