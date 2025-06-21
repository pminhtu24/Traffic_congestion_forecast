import torch
print(torch.__version__)
print(torch.cuda.get_arch_list())  # phải ra như ['sm_86', 'sm_89', ...] nếu GPU được hỗ trợ
print(torch.cuda.is_available())