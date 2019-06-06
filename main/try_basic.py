

from scvi.models.log_likelihood import log_zinb_positive, log_nb_positive
from scvi.models.modules import Encoder, DecoderSCVI, LinearDecoderSCVI
from scvi.models.utils import one_hot

torch.backends.cudnn.benchmark = True


class P1oint(nn.Module):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
    def forward(self,c):
        #print("value is %.3d" %(((self.x ** 2) + (self.y ** 2)) ** c))
        print("value is {:.5f}" .format(((self.x ** 2) + (self.y ** 2)) ** c))
        return(((self.x ** 2) + (self.y ** 2)) ** c)
xx = P1oint(2,4)
ZZ=xx(.5)




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
        xx = torch.mean(x)
        return xx

lay = MyNet()

lay(x)


def myFun(x,**argv):
    for key,arg in argv.items():
        print (arg)
    return([sum(x),x])

l,_=myFun([2,3],first ='Geeks', mid ='for', last='Geeks')

import sys
sys.getsizeof(gene_dataset.X)

# Making an iterator: the object-oriented way

class Count:

    """Iterator that counts upward forever."""

    def __init__(self, start=0):
        self.num = start

    def __iter__(self):
        return self

    def __next__(self):
        num = self.num
        self.num += 1
        return num


tt = [t + 2 for t in [2, 3, 4]]
def print_iterator(it):
    for x in it:
        print(x, end=' ')
    print('')

def myfun(x):
    return(x+2)
ff = map(myfun,[1,4,2])
list(ff)
ff
list_numbers = [1, 2, 3, 4]

map_iterator = map(lambda x: x * 2, list_numbers)
list(map_iterator)

favorite_numbers = [6, 57, 4, 7, 68, 95]


sq = (n**2 for n in favorite_numbers)
next(sq)

for n in sq:
    print(n)


class Count:

    """Iterator that counts upward forever."""

    def __init__(self, start=0):
        self.num = start

    def __iter__(self):
        return self

    def __next__(self):
        num = self.num
        self.num += 1
        return num

c = Count(start = 2)
c.__next__()



def my_gen(n):
    print('This is printed first')
    # Generator function contains yield statements
    yield n

    n += 1
    print('This is printed second')
    yield [n,n]

    n += 1
    print('This is printed at last')
    yield [n,n,n]

xx = my_gen(2)
next(xx)




xx = filter(lambda p: p>0, [2,-3,4,10,-333,4])
list(xx)
next(xx)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence as kl

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.conv2 = nn.Conv2d(4, 9, 3)
        self.full  = nn.Sequential(nn.Sigmoid(), nn.Linear(5, 4))
    def forward(self, x):
       x = F.relu(self.conv1(x))
       x = F.relu(self.conv2(x))
       x = self.full(x)
       return (x)


mo = Model()

dict = mo.state_dict()
dict['conv2.weight'].shape
dict.keys()
[print(key,"-", values.shape) for (key,values) in zip(dict.keys(),dict.values())]
x = torch.randn(1,1,32,32)

xx = mo(x)
xx
param = list(mo.parameters())

len(param)


def init_weights(m):
    print(m)
    #if type(m) == nn.Linear:
    if type(m) == nn.Conv2d:
       m.weight.data.fill_(2.0)
       print(m.weight)
net = nn.Sequential(nn.Linear(2, 2), nn.Sigmoid(), nn.Linear(2, 2))
net.apply(init_weights)
moo = mo.apply(init_weights)

moo.state_dict()["conv1.weight"]

conv1 = nn.Conv2d(1, 4, 3)
myweight = conv1.weight.data.uniform_(10.0,1.0)




my_state_dict = moo.state_dict()
moo = Model()

moo = mo.apply(init_weights)

torch.save(moo.state_dict(), "saved_model/try_dict.pt")
mo = Model()
mo.state_dict()
mo.load_state_dict(torch.load("saved_model/try_dict.pt"))
mo.eval()
