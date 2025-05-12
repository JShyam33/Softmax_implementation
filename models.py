
import torch.nn as nn
class Norm(nn.Module):
    def __init__(self):
        super(Norm, self).__init__()

    def forward(self, x):
        # Normalize manually so that each row sums to 1.
        # A small epsilon is added to prevent division by zero.
        x_sum = x.sum(dim=1, keepdim=True) + 1e-8
        x = x / x_sum
        return x
class SoftmaxApprox(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_prob=0.2):
        """
        Args:
            input_dim (int): Dimension of the input logit vector.
            hidden_dims (list): List of integers specifying the hidden layer sizes.
            dropout_prob (float): Dropout probability for all dropout layers.
        """
        super(SoftmaxApprox, self).__init__()
        layers = []
        in_dim = input_dim
        # Create several blocks of [Dense -> BatchNorm -> ReLU -> Dropout]




        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_dim = h

        self.feature_extractor = nn.Sequential(*layers,nn.Linear(in_dim,input_dim ),nn.ReLU(),Norm())

    def forward(self, x):
        # Pass through the hidden blocks.

        x = self.feature_extractor(x)
        return x

    def freeze(self):
        """
        Freeze all parameters so they won't be updated during training.
        """
        for param in self.parameters():
            param.requires_grad = False

class ExpApprox(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout_prob=0.2):
        """
        Args:
            input_dim (int): Dimension of the input logit vector.
            hidden_dims (list): List of integers specifying the hidden layer sizes.
            dropout_prob (float): Dropout probability for all dropout layers.
        """
        super(ExpApprox, self).__init__()
        layers = []
        in_dim = input_dim
        # Create several blocks of [Dense -> BatchNorm -> ReLU -> Dropout]




        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_dim = h

        self.feature_extractor = nn.Sequential(*layers,nn.Linear(in_dim,input_dim ))

    def forward(self, x):
        # Pass through the hidden blocks.

        x = self.feature_extractor(x)
        return x

    def freeze(self):
        """
        Freeze all parameters so they won't be updated during training.
        """
        for param in self.parameters():
            param.requires_grad = False