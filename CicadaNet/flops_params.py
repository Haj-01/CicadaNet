import torch
import torch.nn as nn
from thop import profile
# from model.test_net6 import TSCNet
# from model.test_net import Net
from model.crn32_glu_tscb import CRN
#

# Model
print('==> Building model..')
model = CRN()

inputs = torch.rand(1, 1, 321, 401)
flops, params = profile(model, (inputs,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))