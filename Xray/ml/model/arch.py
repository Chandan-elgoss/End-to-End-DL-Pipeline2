"""DEPRECATED (legacy CNN for classification).

This module implemented a custom CNN for the earlier Lung X-ray classification
pipeline. For the current YOLO object-detection setup, this file is not used.
Left in the repository for reference; intentionally disabled below.
"""

if False:  # keep code reference without importing heavy deps
    import torch.nn as nn
    import torch.nn.functional as F

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.convolution_block1 = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=(3, 3), padding=0, bias=True),
                nn.ReLU(),
                nn.BatchNorm2d(8),
            )
            # ... rest of legacy architecture ...

        def forward(self, x) -> float:
            # legacy forward pass
            return F.log_softmax(x, dim=-1)