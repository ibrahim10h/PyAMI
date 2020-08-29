# Set the device to be used (CPU or GPU)

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
