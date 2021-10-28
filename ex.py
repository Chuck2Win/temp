import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import argparse
from tqdm import tqdm
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int,default = 1)
args = parser.parse_args()
print(args)
model = nn.Linear(10,10)
x = torch.randn((1000,10))
torch.cuda.set_device(args.local_rank)
torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://')
model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[args.local_rank],
                                                      output_device=args.local_rank)
model.cuda()
for i in tqdm(x):
    model.forward(i.unsqueeze(0).cuda())
print('end')
