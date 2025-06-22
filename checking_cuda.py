import torch
print(torch.__version__)
print(torch.cuda.get_arch_list())  # ['sm_86', 'sm_89', ...]
print(torch.cuda.is_available())