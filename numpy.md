# `numpy` Notes

## What Is `numpy` ?

NumPy (Numerical Python) is a Python library used for fast numerical computation.
It provides:

- A powerful N-dimensional array object (ndarray).
- Tools for array creation, manipulation, indexing, and math operations.
- Functions for linear algebra, statistics, random numbers, etc.
- Core building block for data science, ML, scientific computing.

---

## Efficiency Of Using `numpy` Arrays over Python List

CODE :

```python
# import numpy module
import numpy as np

# import time module
import time

# store the start time of the operation in a var
start_time = time.time()

# create a pythonic list
PythonicList = list(i for i in range(100000000))

# print the time taken by the operation
print(f"time taken by pythonic list : {time.time() - start_time}")

# store the start time of the operation again
start_time = time.time()

# create a numpy array
NumpyArray = np.arange(100000000)

# print the time taken by the operation
print(f"time taken by numpy array   : {time.time() - start_time}")
```

OUTPUT :

```bash
time taken by pythonic list : 4.570127964019775
time taken by numpy array   : 0.1838090419769287
```

---

## To Use `numpy` In Your Python Program First Import It

```python
# import the numpy module
import numpy as np
```

## Creating Arrays Using `numpy.array`

- creating 1D array (vector)
- creating 2D array (matrix)
- creating 3D array (tensor)

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

```bash
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

```bash
tensor :
[[[10 20]
  [30 40]
  [50 60]
  [70 80]]]
```

## Creating Arrays In A Sequence Using `numpy.arange` and `numpy.reshape`

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

```bash
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

## Creating Arrays Of Random Numbers

- creating vectors of random numbers using `numpy.random.random` or `numpy.random.randint`

- creating matrices of random numbers using `numpy.random.random` or `numpy.random.randint`

- creating tensors of random numbers using `numpy.random.random` or `numpy.random.randint`



CODE

```python
# import numpy module
import numpy as np

# create vector of size = 5 randoms floats between 0 and 1
RandomFloatVector = np.random.random(5)

# print the random float vector created
print(RandomFloatVector, "\n\n")

# create a matrix of order 3 x 3 of random floats between 0 and 1
RandomFloatMatrix = np.random.random((3, 3))

# print the random float matrix
print(RandomFloatMatrix, "\n\n")

# create a tensor of order 3 x 3 x 3 of random floats between 0 and 1
RandomFloatTensor = np.random.random((3, 3, 3))

# print the random float tensor created
print(RandomFloatTensor)


```

OUTPUT

```bash
[0.23261866 0.26140708 0.86520425 0.75848118 0.39575855] 


[[0.13919869 0.9769624  0.69334485]
 [0.12177854 0.94113333 0.0273819 ]
 [0.11979926 0.98755775 0.47005629]] 


[[[0.7353939  0.87675499 0.94689317]
  [0.85479062 0.08657959 0.67521377]
  [0.01601717 0.71270791 0.10672972]]

 [[0.72494675 0.43746822 0.60158548]
  [0.03970505 0.55018179 0.08792939]
  [0.15400783 0.85600297 0.47159429]]

 [[0.68214627 0.76015265 0.75723404]
  [0.3966675  0.3929398  0.80938259]
  [0.07578862 0.70283659 0.57201887]]]
```
