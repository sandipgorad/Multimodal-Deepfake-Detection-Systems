import torch.nn as nn


class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Linear(40, 2)

    def forward(self, features):
        return self.classifier(features)

