# `numpy` Notes Phase - II

## Topics Covered

- Creating Arrays In `numpy`
- Creating Arrays In `numpy` Of A Certain Sequence And Dimensions
- Creating Arrays Of Specific Data Types
- Data Type Of `numpy` Arrays
- Creating `numpy` Array Of Random Numbers Between (0 and 1)
- Creating `numpy` Array Of Random Numbers Between A Certain Range
- Creating `numpy` Array Of Equally Spaced Elements
- Creating `numpy` Array Of 0's
- Creating `numpy` Array Of 1's
- Creating `numpy` Array Of A Certain Value
- Creating Identity Matrix Using `numpy`
- Attributes Of `numpy` Arrays
- Changing Data Type Of `numpy` Arrays

---

## 1D Array = vector

## 2D Array = matrix

## 3D Array = Tensor

## Creating `numpy` Arrays Using `numpy.array`

CODE

```python
# import numpy module
import numpy as np

# create a 1D numpy array (vector)
vector = np.array([10, 20, 30, 40, 50])

# print the vector created
print(vector, "\n\n")

# create a 2D numpy array (matrix)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# print the matrix created
print(matrix, "\n\n")

# create a 3D numpy array (tensor)
tensor = np.array([[[1, 2],
                    [3, 4],
                    [5, 6],
                    [7, 8]]])

# print the tensor created
print(tensor, "\n\n")

```

OUTPUT

```zsh
[10 20 30 40 50]


[[1 2 3]
 [4 5 6]
 [7 8 9]]


[[[1 2]
  [3 4]
  [5 6]
  [7 8]]]


```

OUTPUT :

```zsh
tensor :
[[[10 20]
  [30 40]
  [50 60]
  [70 80]]]
```

---

## Creating `numpy` Arrays In A Sequence Using `numpy.arange` and `numpy.reshape`

- creating vectors using `numpy.arange`
- creating matrices using `numpy.arange` and `numpy.reshape`
- creating tensors using `numpy.arange` and `numpy.reshape`

CODE

```python
# import numpy module
import numpy as np

# create a vector of elements 1 - 10
nums = np.arange(1, 11)

# print the created list
print(nums, "\n\n")

# create a matrix of order 5 x 5 of even nums
matrix = np.arange(2, 51, 2).reshape(5, 5)

# print the created matrix
print(matrix, "\n\n")

# create a tensor of order 2 x 2 x 2 of elements 1 - 8
tensor = np.arange(1, 9).reshape(2, 2, 2)

# print the created tensor
print(tensor, "\n\n")
```

OUTPUT

```zsh
[ 1  2  3  4  5  6  7  8  9 10]


[[ 2  4  6  8 10]
 [12 14 16 18 20]
 [22 24 26 28 30]
 [32 34 36 38 40]
 [42 44 46 48 50]]


[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]
```

---

## Creating `numpy` Arrays Of Specific Data Types

CODE

```python
# import numpy module
import numpy as np

# creating a vector of floats
vector = np.array([10, 20, 30, 40, 50], dtype=float)

# print the created vector
print(vector, "\n\n")

# create a matrix of complex
matrix = np.array([[1, 1 ,1 ],
                      [2, 2, 2,],
                      [3, 3, 3]], dtype=complex)

# print the created matrix
print(matrix, "\n\n")

# creating a tensor of type boolean
tensor = np.array([[[0, 1],
                    [1, 0],
                    [0, 0]]], dtype=bool)

# print the created tensor
print(tensor, "\n\n")
```

OUTPUT

```zsh
[10. 20. 30. 40. 50.]


[[1.+0.j 1.+0.j 1.+0.j]
 [2.+0.j 2.+0.j 2.+0.j]
 [3.+0.j 3.+0.j 3.+0.j]]


[[[False  True]
  [ True False]
  [False False]]]


```

---

## Data Type Of `numpy` Arrays

```python
# import numpy module
import numpy as np

# create a numpy array
arr = np.array([10, 20, 30, 40, 50])

# print the datatype of the numpy array
print(type(arr))
```

```zsh
<class 'numpy.ndarray'>
```

## Creating `numpy` Arrays Of Random Numbers (between 0 and 1)

- method used for vectors : `numpy.random.random(size=size)`
- method used for matrices : `numpy.random.random(size=(rows, cols))`
- method used for tensors : `numpy.random.random(size=(rows, cols, depth))`

```python
# import numpy module
import numpy as np

# create vector of size = 5 random int between 1 and 0
RandomFloatVector = np.random.random(size=5)

# print the random vector created
print(RandomFloatVector, "\n\n")

# create a matrix of order 3 x 3 of random int between 1 and 0
RandomFloatMatrix = np.random.random(size=(3, 3))

# print the random int matrix
print(RandomFloatMatrix, "\n\n")

# create a tensor of order 3 x 3 x 3 of random int between 1 and 0
RandomFloatTensor = np.random.random(size=(3, 3, 3))

# print the random int tensor created
print(RandomFloatTensor)


```

OUTPUT

```zsh
[0.50966683 0.91670581 0.36744857 0.56907233 0.21254982]


[[0.83385321 0.02005452 0.68330472]
 [0.17492511 0.30839177 0.12430648]
 [0.11759084 0.75318799 0.99297331]]


[[[0.85801743 0.04244979 0.65939733]
  [0.93197597 0.58581609 0.20292197]
  [0.93918191 0.07438105 0.38702471]]

 [[0.76573969 0.99445224 0.54959985]
  [0.32422686 0.7541187  0.09219974]
  [0.51647366 0.98520193 0.34397012]]

 [[0.21429105 0.49505295 0.29678025]
  [0.54835222 0.66204691 0.70871094]
  [0.44575704 0.14717658 0.62734236]]]
```

---

## Creating `numpy` Arrays Of Random Numbers (between a certain range)

> method used for vectors : `numpy.random.randint(low=low_value, high=high_value, size=size)`  
> method used for matrices : `numpy.random.randint(low=low_value, high=high_value, size=(rows, cols))`  
> method used for tensors : `numpy.random.randint(low=low_value, high=high_value, size=(rows, cols, depth))`

- low → Lowest integer (inclusive).
- high → Highest integer (exclusive).
- If not given, integers are generated between 0 and low.
- size → Shape of output (e.g., 5 -> 1D array of 5 numbers, (2, 3)-> 2D array of total 6 elements , (2, 2, 2) -> 3D array of total 8 elements).
- dtype → Type of output integers (default = int).

  CODE

```python
# import numpy module
import numpy as np

# create vector of size = 5 random int between 1 and 10
RandomFloatVector = np.random.randint(low=1, high=11, size=5)

# print the random vector created
print(RandomFloatVector, "\n\n")

# create a matrix of order 3 x 3 of random int between 1 and 10
RandomFloatMatrix = np.random.randint(low=1, high=11, size=(3, 3))

# print the random int matrix
print(RandomFloatMatrix, "\n\n")

# create a tensor of order 3 x 3 x 3 of random int between 1 and 10
RandomFloatTensor = np.random.randint(low=1, high=11, size=(2, 2, 2))

# print the random int tensor created
print(RandomFloatTensor)

```

OUTPUT

```zsh
[5 8 8 7 6]


[[ 2  2  4]
 [10  4  3]
 [ 7  1  2]]


[[[ 9  5]
  [ 2 10]]

 [[ 8  3]
  [ 5  8]]]
```

---

## Creating `numpy` Arrays Of Equally Spaced Elements

```python
# import numpy module
import numpy as np

# create a vector of equally spaced elements from 2.5 to 10 in steps of 4
vector = np.linspace(2.5, 10.0, 4)

# print the created vector
print(vector, "\n\n")


# create a matrix of order (5 x 5) of equally spaced elements from 5 - 125 in steps of 25
matrix = np.linspace(5, 125, 25).reshape(5, 5)

# print the created vector
print(matrix, "\n\n")
```

OUTPUT

```zsh
[ 2.5  5.   7.5 10. ]


[[  5.  10.  15.  20.  25.]
 [ 30.  35.  40.  45.  50.]
 [ 55.  60.  65.  70.  75.]
 [ 80.  85.  90.  95. 100.]
 [105. 110. 115. 120. 125.]]

```

---

## Creating `numpy` Arrays Of 0's

- creating vector of 0's
- creating matrix of 0's
  CODE

```python
# import numpy module
import numpy as np

# create a vector of 0's of size = 10
ZeroVector = np.zeros(10)

# print the vector of 0's created
print(ZeroVector, "\n\n")

# create a 5 x 5 matrix of 0's
ZeroMatrix = np.zeros(25).reshape(5, 5)

# print the matrix of 0's created
print(ZeroMatrix, "\n\n")
```

OUTPUT

```zsh
[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]


[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]

```

## Creating `numpy` Arrays Of 1's

- creating vector of 1's
- creating matrix of 1's
  CODE

```python
# import numpy module
import numpy as np

# create a vector of 1's of size = 10
OnesVector = np.ones(10)

# print the vector of 1's created
print(OnesVector, "\n\n")

# create a 5 x 5 matrix of 1's
OnesMatrix = np.ones(25).reshape(5, 5)

# print the matrix of 1's created
print(OnesMatrix, "\n\n")
```

OUTPUT

```zsh
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]


[[1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]]

```

---

## Creating `numpy` Arrays of A Scalar Value

- creating vector of a scalar value (= k)
- creating matrix of a scalar value (= k)
  CODE

```python
# import numpy module
import numpy as np

# create a vector of size = 10 of all values = 50
ScalarVector = np.full(10, 50)

# print the scalar vector created
print(ScalarVector, "\n\n")

# create a matrix of 5 x 5 of all values = 4
FoursMatrix = np.full(25, 4).reshape(5, 5)

# print the matrix of 4's
print(FoursMatrix, "\n\n")
```

OUTPUT

```zsh
[50 50 50 50 50 50 50 50 50 50]


[[4 4 4 4 4]
 [4 4 4 4 4]
 [4 4 4 4 4]
 [4 4 4 4 4]
 [4 4 4 4 4]]


```

---

## Creating An Identity Matrix Using `numpy` Arrays

CODE

```python
# import numpy module
import numpy as np

# create an identity matrix of size 3 x 3
ThreeIdentity = np.identity(3)

# print the identity matrix created
print(ThreeIdentity, "\n\n")

# create an identity matrix of size 5 x 5
FiveIdentity = np.identity(5)

# print the identity matrix created
print(FiveIdentity, "\n\n")
```

OUTPUT

```zsh
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]


[[1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 0. 1.]]


```

## Attributes Of `numpy` Arrays

CODE

```python
# import numpy module
import numpy as np

# create a vector of size 5
vector = np.array([10, 20, 30, 40, 50])

# print the vector
print(f"vector : \n{vector}\n")

# print the dimensions of the vector
print(f"dimensions of vector : {vector.ndim}")

# print the shape of the vector
print(f"shape of the vector : {vector.shape}")

# print the size of the vector
print(f"size of the vector is : {vector.size}")

# print the data type of the vector
print(f"data type of the vector is : {vector.dtype}")

# print the item size of the vector
print(f"item size of the vector is : {vector.itemsize}")

# print the total bytes used by the vector
print(f"total bytes used by the vector is : {vector.nbytes}")

```

OUTPUT

```zsh
vector :
[10 20 30 40 50]

dimensions of vector : 1
shape of the vector : (5,)
size of the vector is : 5
data type of the vector is : int64
item size of the vector is : 8
total bytes used by the vector is : 40
```

CODE

```python

```

OUTPUT

```zsh

```

CODE

```python
# import numpy module
import numpy as np

# create a matrix of 3 x 4
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [1, 1, 1]])

# print the matrix
print(f"matrix : \n{matrix}\n")

# print the dimensions of the matrix
print(f"dimensions of matrix : {matrix.ndim}")

# print the shape of the matrix
print(f"shape of the matrix : {matrix.shape}")

# print the size of the matrix
print(f"size of the matrix is : {matrix.size}")

# print the data type of the matrix
print(f"data type of the matrix is : {matrix.dtype}")

# print the item size of the matrix
print(f"item size of the matrix is : {matrix.itemsize}")

# print the total bytes used by the matrix
print(f"total bytes used by the matrix is : {matrix.nbytes}")

```

OUTPUT

```zsh
matrix :
[[1 2 3]
 [4 5 6]
 [7 8 9]
 [1 1 1]]

dimensions of matrix : 2
shape of the matrix : (4, 3)
size of the matrix is : 12
data type of the matrix is : int64
item size of the matrix is : 8
total bytes used by the matrix is : 96

```

CODE

```python
# import numpy module
import numpy as np

# create a tensor of 2 x 2 x 2
tensor = np.array([[[1, 2],
                    [3, 4]],
                   [[5, 6],
                    [7, 8]]])

# print the tensor
print(f"tensor : \n{tensor}\n")

# print the dimensions of the tensor
print(f"dimensions of tensor : {tensor.ndim}")

# print the shape of the tensor
print(f"shape of the tensor : {tensor.shape}")

# print the size of the tensor
print(f"size of the tensor is : {tensor.size}")

# print the data type of the tensor
print(f"data type of the tensor is : {tensor.dtype}")

# print the item size of the tensor
print(f"item size of the tensor is : {tensor.itemsize}")

# print the total bytes used by the tensor
print(f"total bytes used by the tensor is : {tensor.nbytes}")

```

## Changing Data Type Of `numpy` Arrays

OUTPUT

```zsh
tensor :
[[[1 2]
  [3 4]]

 [[5 6]
  [7 8]]]

dimensions of tensor : 3
shape of the tensor : (2, 2, 2)
size of the tensor is : 8
data type of the tensor is : int64
item size of the tensor is : 8
total bytes used by the tensor is : 64

```

CODE

```python
# import numpy module
import numpy as np

# create a tensor of 2 x 2 x 2
tensor = np.array([[[1, 0],
                    [0, 4]],
                   [[0, 6],
                    [0, 8]]])

# print the tensor
print(f"tensor : \n{tensor}\n")

# convert the data type of tensor to float
new_tensor = tensor.astype(np.int8)

# print the new tensor
print(f"new tensor : \n{new_tensor}\n")
```

OUTPUT

```zsh
tensor :
[[[1 0]
  [0 4]]

 [[0 6]
  [0 8]]]

new tensor :
[[[1 0]
  [0 4]]

 [[0 6]
  [0 8]]]

```

---
