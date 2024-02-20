import sys
import torch
import numpy as np
# # from pynvml import *
#
#
# a = np.random.randint(0, 10, (2, 5, 3, 6, 100))
# max_ = np.amax(a, axis=(3, 4), keepdims=False)
# print(max_.shape)
#
#
# # # a = torch.randint(-1000, 1000, (10000, 10000)).to("cuda:0")
# # divisor = 1024*1024*1024
# # # print(sys.getsizeof(a.storage())/divisor)
# #
# #
# # # asize = a.element_size() * a.nelement()
# # # print(asize/divisor)
# # # t = torch.cuda.get_device_properties(0).total_memory
# # # r = torch.cuda.memory_reserved(0)
# # # a = torch.cuda.memory_allocated(0)
# # # f = r-a  # free inside reserved
# # # print(t/divisor, r/divisor, a/divisor, f/divisor)
# # nvmlInit()
# # h = nvmlDeviceGetHandleByIndex(0)
# # info = nvmlDeviceGetMemoryInfo(h)
# # print(f'total    : {info.total/divisor}')
# # print(f'free     : {info.free/divisor}')
# # print(f'used     : {info.used/divisor}')
# # # print(torch.cuda.mem_get_info(device="cuda:0"))
#
# # for item in range(0, 100+1, 100):
# #     print(item)

import pandas as pd
import numpy as np

l1 = ['one', 'two', 'three']
l2 = [ii for ii in range(1,4)]
d = {'col1': l1, 'col2': l2}
df = pd.DataFrame(data=d)
print(df)



