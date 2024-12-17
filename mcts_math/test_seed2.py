import torch 

def get_value_estimate():
    get_value = []
    for i in range(10):
        get_value.append(torch.rand(1))
    return get_value
        