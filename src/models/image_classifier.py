import torch.nn as nn


class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Linear(2048, 2)

    def forward(self, features):
        return self.classifier(features)