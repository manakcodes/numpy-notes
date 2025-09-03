# `numpy` Notes Phase - VII

## Topics Covered

- Broadcasting In `numpy` Arrays

---

### Broadcasting In `numpy` Arrays

Broadcasting in `numpy` arrays refer to how `numpy` treats the arrays of different shapes during arithmetic operations.

#### Rules Of Broadcasting

1. Make the two arrays have the same number of dimensions.
   if the number of dimensions of the two arrays are different, add new dimensions with size 1 to the head (or to the front) of the array with the smaller dimension.

   **example - 1 :**  
    matrix dimensions = (rows = 2, cols = 2)  
    vector dimensions = (size = 2,)

   then to do any arithmetic operation between matrix and vector first the dimensions of the vector would be changed to (new dimensions of vector = (row = 1, cols = 3))

   **example - 2 :**  
    tensor dimensions = (rows = 2, cols = 2, depth = 2)  
    vector dimensions = (size = 2,)

   then to do any arithmetic operation between tensor and vector first the dimensions of the vector would be changed to (new dimensions of vector = (row = 1, cols = 1, depth = 3))

2. Make each dimension of the two arrays the same size

- if the sizes of the two arrays do not match, dimensions with size = 1 are stretched to the size of the other array.

**example - 1 :**  
 matrix dimensions = (rows = 3, cols = 3)  
 vector dimensions = (size = 2,)

then to do any arithmetic operation between matrix and vector first the dimensions of the vector would be changed to (new dimensions of vector = (row = 1, cols = 3)), then the dimensions that are changed to 1 are stretched to the dimensions of the array of larger dimensions (in this case to 3).

**example - 2 :**  
 tensor dimensions = (rows = 2, cols = 2, depth = 2)  
 vector dimensions = (size = 2,)

then to do any arithmetic operation between tensor and vector first the dimesnions of the vector would be changed to (new dimensions of vector = (row = 1, cols = 1, depth = 3)) then the dimensions that are changed to 1 are stretched to the dimensions of the array of larger dimensions (in this case to 2, 2).

- if there is a dimensions whose size is not 1 in either of the two arrays, it then cannot be broadcasted, and an error is raised.

(please make sure to try the following code - output snippets on pen and paper at least once according to the rules of broadcasting to get a better grasp of concept)

CODE

```python
# import numpy module
import numpy as np

# create a matrix
matrix = np.arange(12).reshape(4, 3)

# create a vector
vector = np.arange(3)

# print the vector and the matrix
print(vector, "\n\n", matrix, "\n\n")

# print the sum of the vector and matrix
print("sum : \n", vector + matrix)
```

OUTPUT

```zsh
[0 1 2]

 [[ 0  1  2]
 [ 3  4  5]
 [ 6  7  8]
 [ 9 10 11]]


sum :
 [[ 0  2  4]
 [ 3  5  7]
 [ 6  8 10]
 [ 9 11 13]]
```

CODE

```python
# import numpy module
import numpy as np

# create a matrix
matrix = np.arange(12).reshape(3, 4)

# create a vector
vector = np.arange(3)

# print the vector and the matrix
print(vector, "\n\n", matrix, "\n\n")

# print the sum of the vector and matrix
print("sum : \n", vector + matrix)
```

OUTPUT

```zsh
[0 1 2]

 [[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]


Traceback (most recent call last):
  File "/Users/manaksingh/Desktop/numpy-notes/test.py", line 14, in <module>
    print("sum : \n", vector + matrix)
                      ~~~~~~~^~~~~~~~
ValueError: operands could not be broadcast together with shapes (3,) (3,4)
```

CODE

```python
# import numpy module
import numpy as np

# create a matrix
MatrixOne = np.arange(3).reshape(3, 1)

# create another matrix
MatrixTwo = np.arange(3).reshape(1, 3)

# print the MatrixOne and the MatrixTwo
print(MatrixOne, "\n\n", MatrixTwo, "\n\n")

# print the sum of the MatrixOne and MatrixTwo
print("sum : \n", MatrixOne + MatrixTwo)
```

OUTPUT

```zsh
[[0]
 [1]
 [2]]

 [[0 1 2]]


sum :
 [[0 1 2]
 [1 2 3]
 [2 3 4]]
```

CODE

```python
# import numpy module
import numpy as np

# create a matrix
MatrixOne = np.arange(3).reshape(1, 3)

# create another matrix
MatrixTwo = np.arange(4).reshape(4, 1)

# print the MatrixOne and the MatrixTwo
print(MatrixOne, "\n\n", MatrixTwo, "\n\n")

# print the sum of the MatrixOne and MatrixTwo
print("sum : \n", MatrixOne + MatrixTwo)
```

OUTPUT

```zsh
[[0 1 2]]

 [[0]
 [1]
 [2]
 [3]]


sum :
 [[0 1 2]
 [1 2 3]
 [2 3 4]
 [3 4 5]]
```

CODE

```python
# import numpy module
import numpy as np

# create a vector
vector = np.array([1])

# create a matrix
matrix = np.arange(4).reshape(2, 2)

# print the vector and the matrix
print(vector, "\n\n", matrix, "\n\n")

# print the sum of the vector and matrix
print("sum : \n", vector + matrix)
```

OUTPUT

```zsh
[1]

 [[0 1]
 [2 3]]


sum :
 [[1 2]
 [3 4]]
```

---
