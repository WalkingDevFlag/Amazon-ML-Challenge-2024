# Checking paddle
# import paddle
# print(paddle.utils.run_check())


# Checking Cuda 
# import torch
# print("Number of GPU: ", torch.cuda.device_count())
# print("GPU Name: ", torch.cuda.get_device_name())
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)


# import paddle
# print(paddle.utils.run_check())        # Should output True
# print(paddle.is_compiled_with_cuda())  # Should output True
# print(paddle.device.get_device())      # Should show GPU info