import torch
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import Encryptor
from torch import nn
import torch.distributions as dist
import torch.nn.functional as F


def entropy(x, axis=0):
    # Ensure probabilities sum to 1 along the specified axis
    probabilities_norm = F.softmax(x, dim=axis)
    # print(probabilities_norm.shape)
    # Create a categorical distribution
    categorical_dist = dist.Categorical(probs=probabilities_norm)
    # print(categorical_dist.entropy().shape)
    # Compute entropy along the specified axis
    entropy_value = categorical_dist.entropy()

    return entropy_value


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

mnist_dataset = datasets.MNIST('./', train=False, transform=transform)
train_size = int(0.8 * len(mnist_dataset))
test_size = len(mnist_dataset) - train_size
mnist_train_dataset, mnist_test_dataset = random_split(mnist_dataset,
                                                       [train_size, test_size])

batch_size = 5000

train_dataloader = DataLoader(mnist_train_dataset, batch_size,
                              shuffle=True, drop_last=True,
                              pin_memory=True)
test_dataloader = DataLoader(mnist_test_dataset, batch_size,
                             shuffle=True, drop_last=True,
                             pin_memory=True)
model = Encryptor(num_blocks=50).to('cuda:0')


def xy_loss(x, y):
    x_cpu = x.flatten(1, -1)
    y_cpu = y.flatten(1, -1)
    return (entropy(x_cpu, axis=1) +
            entropy(y_cpu, axis=1) -
            entropy(torch.cat((x_cpu, y_cpu), dim=1), axis=1)).mean()


def conditional_mutual_information(x, y, z):
    x_cpu = x.flatten(1, -1)
    y_cpu = y.flatten(1, -1)
    z_cpu = z.flatten(1, -1)

    xyz = torch.cat((x_cpu, y_cpu, z_cpu), dim=1)
    h_xyz = entropy(xyz, axis=1)

    h_i = entropy(z_cpu, axis=1) + entropy(x_cpu, axis=1) + entropy(y_cpu,
                                                                    axis=1)
    h_ij = entropy(torch.cat((x_cpu, y_cpu), dim=1), axis=1) +\
        entropy(torch.cat((z_cpu, y_cpu), dim=1), axis=1) +\
        entropy(torch.cat((x_cpu, z_cpu), dim=1), axis=1)
    # print(h_xyz - h_ij + h_i)
    return (h_xyz - h_ij + h_i).mean()


mse = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in tqdm(range(100)):
    for image, _ in train_dataloader:
        image = image.to('cuda:0').requires_grad_(requires_grad=True)
        key = torch.randn(size=(batch_size, 784))\
            .to(device='cuda:0', dtype=torch.float32)\
            .requires_grad_(requires_grad=True)
        optimizer.zero_grad()
        # print(image, key)
        cyphroimage = model(image, key)
        decoded_img = model(cyphroimage, key, reverse=True)
        mse_loss = mse(image, decoded_img)
        encryption_loss = xy_loss(image, cyphroimage)
        decryption_loss = F.relu(6 -
                                 conditional_mutual_information(cyphroimage,
                                                                image, key))
        loss = mse_loss + encryption_loss + decryption_loss
        loss.backward()
        optimizer.step()
    print(f'mse = {mse_loss}, enc = {encryption_loss}, dec= {decryption_loss}')
