import torch.nn as nn
import torchvision.models as models


class SimCLRModel(nn.Module):

    #resnet18 encoder + mlp projection head
    #use encoder features only after training
    
    def __init__(self, feature_dim=128):
        
        super().__init__()

        resnet = models.resnet18(weights=None)

        #3x3 conv preserves spatial info for cifar
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        #remove maxpool, too aggressive for 32x32
        resnet.maxpool = nn.Identity()

        self.encoder_dim = resnet.fc.in_features  #512
        resnet.fc = nn.Identity()
        self.encoder = resnet

        #512 512 128 contrastive pretraining only
        self.projection_head = nn.Sequential(
            nn.Linear(self.encoder_dim, self.encoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.encoder_dim, feature_dim)
        )


    def forward(self, x):

        features = self.encoder(x)
        projections = self.projection_head(features)

        return features, projections

    def get_features(self, x):
        return self.encoder(x)
