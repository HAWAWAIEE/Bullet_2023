import torch
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name(0))

if torch.cuda.is_available():
    tensor = torch.rand(3, 3)
    tensor = tensor.to('cuda')
    print(tensor)
