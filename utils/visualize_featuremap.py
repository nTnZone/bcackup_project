import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.torch_utils import intersect_dicts
import os
import sys
import yaml
from models.yolo import Model
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import matplotlib.pyplot as plt


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(3, 6, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2)

        self.conv_trans1 = nn.ConvTranspose2d(6, 3, 4, 2, 1)
        self.conv_trans2 = nn.ConvTranspose2d(3, 1, 4, 2, 1)

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.conv_trans1(x))
        x = self.conv_trans2(x)
        return x


model = MyModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# epochs = 1
# for epoch in range(epochs):
#     for batch_idx, (data, target) in enumerate(loader):
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, data)
#         loss.backward()
#         optimizer.step()
#
#         print('Epoch {}, Batch idx {}, loss {}'.format(
#             epoch, batch_idx, loss.item()))


def normalize_output(img):
    img = img - img.min()
    img = img / img.max()
    return img


# Plot some images
# idx = torch.randint(0, output.size(0), ())
# pred = normalize_output(output[idx, 0])
# img = data[idx, 0]

# fig, axarr = plt.subplots(1, 2)
# axarr[0].imshow(img.detach().numpy())
# axarr[1].imshow(pred.detach().numpy())

# Visualize feature maps
activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()

    return hook


# model.conv1.register_forward_hook(get_activation('conv1'))
# data, _ = dataset[0]
# data.unsqueeze_(0)
# output = model(data)
#
# act = activation['conv1'].squeeze()
# fig, axarr = plt.subplots(act.size(0))
# for idx in range(act.size(0)):
#     axarr[idx].imshow(act[idx])
#
# kernels = model.conv1.weight.detach()
# fig, axarr = plt.subplots(kernels.size(0))
# for idx in range(kernels.size(0)):
#     axarr[idx].imshow(kernels[idx].squeeze())
#
# from torchvision.utils import make_grid
#
# kernels = model.extractor[0].weight.detach().clone()
# kernels = kernels - kernels.min()
# kernels = kernels / kernels.max()
# img = make_grid(kernels)
# plt.imshow(img.permute(1, 2, 0))

# --------------------------------------------
# Model
device = 'cuda'
weights = os.path.join(sys.path[1], 'yolov5/best.pt')
nc = 4  # num of class
with open('hyp.yaml') as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)  # load hyps

ckpt = torch.load(weights, map_location=device)  # load checkpoint
model = Model(ckpt['model'].yaml, ch=3, nc=nc, anchors=hyp.get('anchors')).to(device)  # create
exclude = ['anchor'] if ( hyp.get('anchors')) and not opt.resume else []  # exclude keys
state_dict = ckpt['model'].float().state_dict()  # to FP32
state_dict = intersect_dicts(state_dict, model.state_dict(), exclude=exclude)  # intersect
model.load_state_dict(state_dict, strict=False)  # load
# logger.info('Transferred %g/%g items from %s' % (len(state_dict), len(model.state_dict()), weights))  # report

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        module.register_forward_hook(get_activation(name))
        print('1')



