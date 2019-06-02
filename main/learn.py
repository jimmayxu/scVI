

from __future__ import print_function
import torch

# https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html




x = torch.ones(1,4 , requires_grad=True)

y = x + 2

print(y)

c = torch.rand(4, 6)
z = y.mm(c)
out = z.prod()

print(z, out)

out.backward()
x.grad

with torch.no_grad():
    print((x ** 2).requires_grad)



#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
import torch
import torch.nn as nn
import torch.nn.functional as F



class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel (32- (5+1)) = 26
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 90)
        self.fc3 = nn.Linear(90, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = MyNet()

params = list(net.parameters())
print(len(params))

print(params[8].size())

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)


params = list(net.parameters())
len(params)
params[0].size()

input = torch.randn(1, 1, 32, 32)
input = input.to(device)

target = 10*torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
target = target.to(device)
criterion = nn.MSELoss()


## -----------------------------
print(loss.grad_fn)
net.zero_grad()

print('conv1.bias.grad before backward')
print(net.conv1.parameters())

loss.backward()

print('conv1.bias.grad after backward')
print(net.fc1.weight)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)
## ----------------------------------

import torch.optim as optim
net = MyNet()

#net = nn.DataParallel(net)
#net.to(device)

params = list(net.parameters())
print(len(params))

print(params[3].size())

input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)


# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

for epoch in range(50): # Does the update
    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers
    output = net(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    output = net(input)
    print('%d epoch, the loss: %f' % (epoch,loss.item()))


"""
https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Parameters and DataLoaders
input_size = 5
output_size = 2

batch_size = 30
data_size = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len

rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size),
                         batch_size=batch_size, shuffle=True)
class Model(nn.Module):
    # Our model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(),
              "output size", output.size())

        return output

model = Model(input_size, output_size)
model = nn.DataParallel(model)
model.to(device)
for data in rand_loader:
    input = data.to(device)
    output = model(input)
    print("Outside: input size", input.size(),
          "output_size", output.size())
