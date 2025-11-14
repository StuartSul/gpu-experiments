# Game plans (implementation order)
# 
# 1. (Python) Create an empty global tensor of 128 + int32
# 2. Fill in the first 128 section of TM with values 0, 1, 2, .. (row major), no packing
# 3. Load from the first 128 section of TM into registers
# 4. Naively save the register values to the empty global tensor
# 5. (Python) Check that empty global tensor has values 0, 1, 2, ...
#
# --> Now we have a way to validate the values on TM. 
#     So we can move on to experimenting with tcgen05.cp

import torch

from _C import kernel

t = torch.zeros(128, dtype=torch.int32, device="cuda")

kernel(t)
torch.cuda.synchronize()

print(t)
