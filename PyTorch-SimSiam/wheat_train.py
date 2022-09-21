import torch
from torch.nn import Identity
from torch.optim import Adam
from torchvision.models import resnet50

from data import create_simsiam_dataloader
from model import SimSiamModel
from train import train
import csv



# The encoder can be any model that returns a feature vector
encoder = resnet50()
encoder.fc = Identity()

model = SimSiamModel(encoder=encoder,
                     out_dim=2048,
                     prediction_head_hidden_dim=512)

optimizer = Adam(params=model.parameters(),
                 lr=4e-4)

train_dataloader = create_simsiam_dataloader(path='train',
                                             valid_exts=['jpeg', 'jpg', 'JPG'],
                                             size=224,
                                             normalize=True,
                                             batch_size=32,
                                             num_workers=8)
valid_dataloader = create_simsiam_dataloader(path='valid',
                                             valid_exts=['jpeg', 'jpg', 'JPG'],
                                             size=224,
                                             normalize=True,
                                             batch_size=32,
                                             num_workers=8)

train(train_dataloader=train_dataloader,
      valid_dataloader=valid_dataloader,
      model=model,
      optimizer=optimizer,
      n_epochs=100)