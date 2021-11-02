import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import os
import json
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# Dataset do složky DataSet -- Dataset1
#                           |- Dataset2

# budu dávat asi jsony které budou obsahovat cestu k obrázku src, dst, a matici

# Rozdělit Dataset na 2-3 Train, Test, Validation 
# Validation je až poslední

# shuffle data

# balance data nemít 40% stejnou ground truth (tohle je asi u clasifierů)


class data(Dataset):

    # the function is initialised here
    def __init__(self, path):
        self.path = path  # check if exists
        self.files = os.listdir(path)

    # the function returns length of data
    def __len__(self):
        return len(self.files)

    # gives one item at a time
    def __getitem__(self, index):
        # OPEN JSON -> LOAD IMAGE PATHS | probably set to grayscale
        #           -> LOAD GROUND TRUTH | probably change format
        filename = self.files[index]
        DatasetItem = json.load(open(os.path.join(self.path, filename)))

        convert_tensor = transforms.ToTensor()
        src = convert_tensor(Image.open(DatasetItem['src'])) 
        dst = convert_tensor(Image.open(DatasetItem['dst'])) 
        groundTruth = torch.tensor(DatasetItem['transformation'])

        stacked = torch.stack((src, dst))
        # print(groundTruth)
        # print(stacked.shape)
        return stacked, groundTruth


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(2, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)

        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1)

        self.conv5 = nn.Conv2d(64, 64, 3, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1)

        self.conv7 = nn.Conv2d(64, 64, 3, 1)
        self.conv8 = nn.Conv2d(64, 64, 3, 1)

        self.fc9 = nn.Linear(16588800, 1024)

        self.fc10 = nn.Linear(1024, 6)

    def forward(self, x):

        # two conv layers
        x = self.conv1(x)
        x = self.conv2(x)

        # max pooling
        x = F.max_pool2d(x, 2)

        # two conv layers
        x = self.conv3(x)
        x = self.conv4(x)

        # max pooling
        x = F.max_pool2d(x, 2)

        # two conv layers
        x = self.conv5(x)
        x = self.conv6(x)

        # max pooling
        x = F.max_pool2d(x, 2)

        # two conv layers
        x = self.conv7(x)
        x = self.conv8(x)

        # two conv layers
        x = self.fc9(x)
        x = self.fc10(x)
        
        return x

dataset = data('.\\..\\Data\MachineData')
dataset.__getitem__(5)
# torch.utils.data.random_split(dataset, lengths)

net = Net()
print(net)
