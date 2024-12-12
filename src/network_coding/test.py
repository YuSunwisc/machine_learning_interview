import torch
import torch.nn as nn


class BertModel(nn.Module):
    def __init__(self):
        # super().__init__()
        super(BertModel, self).__init__()
        print('works')

bert = BertModel()
bert