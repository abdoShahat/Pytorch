# Introduction to deep learning with PyTorch

# Pytorch:-
- Pytorch Is:
    -  one of the most popular deep learning frameworks
    -  the framework used in many published deep learning papers
    -  intuitive and user-friendly
    -  has much in common with NumPy

# Importing PyTorch and related packages
-  PyTorch import in Python

    ```python
    import torch 
    ```
- PyTorch supports
    - image data with **torchvision**
    - audio data with **torchaudio**
    - text data with **torchtext**

# Tensors: the building blocks of networks in PyTorch

- Load from list
```python
import torch 
lst = [[1, 2, 3], [4, 5, 6]] 
tensor = torch.tensor(lst)
```

- Load from NumPy array
```python
import torch 
np_array = np.array(array) 
np_tensor = torch.from_numpy(np_array)
```

# Tensor attributes

- Tensor shape
```python
lst = [[1, 2, 3], [4, 5, 6]] 
tensor = torch.tensor(lst) 
tensor.shape 
```
output
```sql
torch.Size([2, 3]) 
```

- Tensor data type
```python
tensor.dtype 
```
output
```sql
torch.int64 
```

# Getting started with tensor operations
-  Compatible shapes
```python
a = torch.tensor([[1, 1],  
                [2, 2]]) 

b = torch.tensor([[2, 2],  
                [3, 3]]) 

a+b 
```

output
```sql
tensor([[3, 3], 
        [5, 5]]) 
```

# Getting started with tensor operations
- Element-wise multiplication
```python
a = torch.tensor([[1, 1],  
                 [2, 2]]) 
b = torch.tensor([[2, 2],  
                 [3, 3]]) 
a * b
```

output
```sql
tensor([[2,  2], 
        [6, 6]]) 
```


# Our first neural network
