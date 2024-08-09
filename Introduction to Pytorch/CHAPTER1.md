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

```python
import torch.nn as nn 
## Create input_tensor with 3 features
input_tensors = torch.tensor(
    [[0.3471, 0.4547, -0.2356]]
) 
```
A linear layer takes an input, applies a linear funciton, and return output 
```python
# Define our first linear layer 
linear_layer = nn.Linear(in_features=3,out_features=2)
# Pass input through linear layer 
output = linear_layer(input_tensor) 
print(output) 
```

output
```sql
tensor([[-0.2415, -0.1604]],  
    grad_fn=<AddmmBackward0>)
```

# Getting to know the linear layer operation
- Each linear layer has a **.weight** and **.bias** property 

```python
output = linear_layer(input_tensor) 
```

- for input **X** , weights **W0** and bias **b0**, the linear layer perform 

y0 = W0.X+b0

in pytorch 

```python
output = W0 @ input + b0 
```

- Weights and biases are initialized randomly
- They are not useful until they are tuned


# Stacking layers with nn.Sequential()

```python
# Create network with three linear layers 
model = nn.Sequential(
    nn.Linear(10,18),
    nn.Linear(18,20),
    nn.Linear(20,5)
)

input_tensor = torch.tensor([[-0.0014,  0.4038,  1.0305,  0.7521, 0.7489, -0.3968,  0.0113, -1.3844, 0.8705, -0.9743]])

# Pass input_tensor to model to obtain output 
output_tensor = model(input_tensor)
print(output_tensor)
```
output
```sql
tensor([[-0.0254, -0.0673,  0.0763,   
        0.0008,  0.2561]], grad_fn=<AddmmBackward0>) 
```

- we obtain output 1 X 5 dimensions
- Output is Still not yet meaningful

- Stacked linear operations
    - We have only seen linear layer networks
    - Each linear layer multiplies its respective
      input with layer weights and adds biases
    - Even with multiple stacked linear layers, output still has linear relationship with input

# Why do we need activation functions?
- **Activation functions** add **non-linearity** to the network
- Model can learn more **complex** relationships with non-linearity

- Sigmiod Function: 
    - Using for Binary Classification Task 

![Sigmoid Function](/images/Sigmoid.PNG)

```python
import torch 
import torch.nn as nn 

input_tensor = torch0tensr([[6.0]])
Sigmoid = nn.Sigmoid()
output = Sigmoid(input_tensor)
```

output 
```sql
tensor([[0.9975]]) 
```

- Activation function as the last layer
```python 
model = nn.Sequential(
    nn.Linear(6,4)  # First Linear Layer
    nn.Linear(4,1)  # Second Linear Layer
    nn.Sigmoid() 
)
```
**Note**. Sigmoid as last step in network of linear layers is **equivalent** to traditional logistic regression. 

# Softmax 

![Softmax Function](/images/Softmax.PNG)

