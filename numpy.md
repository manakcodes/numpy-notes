# `numpy` Notes

## What Is `numpy` ?

NumPy (Numerical Python) is a Python library used for fast numerical computation.
It provides:

- A powerful N-dimensional array object (ndarray).
- Tools for array creation, manipulation, indexing, and math operations.
- Functions for Linear Algebra, Statistics, Random Numbers, etc.
- Core building block for Data Science, Machine Learning, Scientific Computing.

---

## To Use `numpy` In Your Python Program First Import It

```python
# import the numpy module
import numpy as np
```

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

```zsh
time taken by pythonic list : 4.570127964019775
time taken by numpy array   : 0.1838090419769287
```

---

## Type Of `numpy` Arrays

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

---

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

## Creating `numpy` Arrays Of Random Numbers (between 0 and 1)

- creating vectors of random numbers using `numpy.random.random` or `numpy.random.random`

- creating matrices of random numbers using `numpy.random.random` or `numpy.random.random`

- creating tensors of random numbers using `numpy.random.random` or `numpy.random.random`

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

```zsh
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

## Creating `numpy` Arrays Of Random Numbers (between a certain range)

- creating vectors of random numbers using `numpy.random.random` or `numpy.random.randint`

- creating matrices of random numbers using `numpy.random.random` or `numpy.random.randint`

- creating tensors of random numbers using `numpy.random.random` or `numpy.random.randint`
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
FiveIdentity = np.identity(3)

# print the identity matrix created
print(FiveIdentity, "\n\n")
```

OUTPUT

```zsh
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]


[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]


```

## Vector Operations On `numpy` Matrices

CODE

```python
# import numpy module
import numpy as np

# create a matrix
MatrixOne = np.array([[1, 1 ,1 ],
                      [2, 2, 2,],
                      [3, 3, 3]])

# print the created matrix
print(MatrixOne, "\n\n")

# create another matrix
MatrixTwo = np.array([[1, 1 ,1 ],
                      [2, 2, 2,],
                      [3, 3, 3]])

# print the created matrix
print(MatrixTwo, "\n\n")

# print the addition of both the matrices
print(MatrixOne + MatrixTwo, "\n\n")

# print the subtraction of both the matrices
print(MatrixOne - MatrixTwo, "\n\n")

# print the element wise multiplication of both the matrices
print(MatrixOne * MatrixTwo, "\n\n")

# print the dot product (matrix multiplication of both the matrices)
print(np.dot(MatrixOne, MatrixTwo), "\n\n")

# print the element wise division of both the matrices
print(MatrixOne / MatrixTwo, "\n\n")

# print the element wise modulo of both the matrices
print(MatrixOne % MatrixTwo, "\n\n")

# print the MatrixOne elements raised to power of elements of MatrixTwo
print(MatrixOne ** MatrixTwo, "\n\n")
```

OUTPUT

```zsh
[[1 1 1]
 [2 2 2]
 [3 3 3]]


[[1 1 1]
 [2 2 2]
 [3 3 3]]


[[2 2 2]
 [4 4 4]
 [6 6 6]]


[[0 0 0]
 [0 0 0]
 [0 0 0]]


[[1 1 1]
 [4 4 4]
 [9 9 9]]


[[ 6  6  6]
 [12 12 12]
 [18 18 18]]


[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]


[[0 0 0]
 [0 0 0]
 [0 0 0]]


[[ 1  1  1]
 [ 4  4  4]
 [27 27 27]]

```

## Scalar Operations On `numpy` Matrices

CODE

```python
# import numpy module
import numpy as np

# create a matrix
MatrixOne = np.array([[1, 1 ,1 ],
                      [2, 2, 2,],
                      [3, 3, 3]])

# print the matrix by adding a scalar k = 2 to it
print(MatrixOne + 2, "\n\n")

# print the created matrix
print(MatrixOne, "\n\n")

# print the transpose of the matrix
print(np.transpose(MatrixOne))



```

OUTPUT

```zsh
# import numpy module
import numpy as np

# create a matrix
MatrixOne = np.array([[1, 1 ,1 ],
                      [2, 2, 2,],
                      [3, 3, 3]])

# print the matrix by adding a scalar k = 2 to it
print(MatrixOne + 2, "\n\n")

# print the created matrix
print(MatrixOne, "\n\n")

# print the transpose of the matrix
print(np.transpose(MatrixOne))



```

## Relational Checking In `numpy` Matrices

CODE

```python
# import numpy module
import numpy as np

# create a matrix
MatrixOne = np.array([[1, 1 ,1 ],
                      [2, 2, 2,],
                      [3, 3, 3]])

# print the created matrix
print(MatrixOne, "\n\n")

# check if the entries of the created matrix are equal to 2
print("== 2\n", MatrixOne == 2, "\n\n")

# check if the entries of the created matrix is greater than equal to 2
print(">= 2\n", MatrixOne >= 2, "\n\n")
```

OUTPUT

```zsh
[[1 1 1]
 [2 2 2]
 [3 3 3]]


== 2
 [[False False False]
 [ True  True  True]
 [False False False]]


>= 2
 [[False False False]
 [ True  True  True]
 [ True  True  True]]

```

## Statistical Functions In `numpy`

- `numpy.min`, `numpy.max`, `numpy.sum`, `numpy.mean`, `numpy.mode`, `numpy.median`, `numpy.std`, `numpy.var`, `numpy.cumsum`

- for row wise operations set _`axis=1`_ (row wise : left to right)
- for column wise operations set _`axis=0`_ (column wise : top to bottom)
- the default value of _`axis`_ is 0 (column wise)

## Row - Wise Operations

CODE

```python
# import numpy module
import numpy as np

# create a matrix of order 3 x 3 of random values between 1 and 10
matrix = np.random.randint(low=1, high=10, size=(3, 3))

# print the random matrix created
print(matrix)

# print the min, max, sum, mean, median, std dev, var, cumsum (row wise)
print("min                 : ", np.min(matrix, axis=1))
print("max                : ", np.max(matrix, axis=1))
print("sum                : ", np.sum(matrix, axis=1))
print("mean               : ", np.mean(matrix, axis=1))
print("median             : ", np.median(matrix, axis=1))
print("standard deviation : ", np.std(matrix, axis=1))
print("variance           : ", np.var(matrix, axis=1))
print("cumulative sum     : ", np.var(matrix, axis=1))
```

OUTPUT

```zsh
[[9 6 8]
 [3 1 4]
 [5 2 8]]
min                 :  [6 1 2]
max                :  [9 4 8]
sum                :  [23  8 15]
mean               :  [7.66666667 2.66666667 5.        ]
median             :  [8. 3. 5.]
standard deviation :  [1.24721913 1.24721913 2.44948974]
variance           :  [1.55555556 1.55555556 6.        ]
cumulative sum     :  [1.55555556 1.55555556 6.        ]
```

## Column Wise Operations

CODE

```python
# import numpy module
import numpy as np

# create a matrix of order 3 x 3 of random values between 1 and 10
matrix = np.random.randint(low=1, high=10, size=(3, 3))

# print the random matrix created
print(matrix)

# print the min, max, sum, mean, median, std dev, var, cumsum (column wise)
print("min                 : ", np.min(matrix, axis=0))
print("max                : ", np.max(matrix, axis=0))
print("sum                : ", np.sum(matrix, axis=0))
print("mean               : ", np.mean(matrix, axis=0))
print("median             : ", np.median(matrix, axis=0))
print("standard deviation : ", np.std(matrix, axis=0))
print("variance           : ", np.var(matrix, axis=0))
print("cumulative sum     : ", np.var(matrix, axis=0))
```

OUTPUT

```zsh
[[1 4 9]
 [9 5 6]
 [4 7 6]]
min                 :  [1 4 6]
max                :  [9 7 9]
sum                :  [14 16 21]
mean               :  [4.66666667 5.33333333 7.        ]
median             :  [4. 5. 6.]
standard deviation :  [3.29983165 1.24721913 1.41421356]
variance           :  [10.88888889  1.55555556  2.        ]
cumulative sum     :  [10.88888889  1.55555556  2.        ]
```

## Using `numpy` Trignonometrix Functions

CODE

```python
# import numpy module
import numpy as np

# create a matrix of order 3 x 3 of random values between 1 and 10
matrix = np.random.randint(low=1, high=10, size=(3, 3))

# print the random matrix created
print(matrix, "\n\n")

# print the sin matrix
print("sin matrix : \n", np.sin(matrix), "\n")

# print the cos matrix
print("cos matrix : \n", np.cos(matrix), "\n")

# print the tan matrix
print("tan matrix : \n", np.tan(matrix), "\n")

# print the cosec matrix
print("cosec matrix : \n", 1 / (np.sin(matrix)), "\n")

# print the sec matrix
print("sec matrix : \n", 1 / (np.cos(matrix)), "\n")

# print the cot matrix
print("cot matrix : \n", 1 / (np.tan(matrix)), "\n")
```

OUTPUT

```zsh
[[5 7 8]
 [9 5 3]
 [4 8 7]]


sin matrix :
 [[-0.95892427  0.6569866   0.98935825]
 [ 0.41211849 -0.95892427  0.14112001]
 [-0.7568025   0.98935825  0.6569866 ]]

cos matrix :
 [[ 0.28366219  0.75390225 -0.14550003]
 [-0.91113026  0.28366219 -0.9899925 ]
 [-0.65364362 -0.14550003  0.75390225]]

tan matrix :
 [[-3.38051501  0.87144798 -6.79971146]
 [-0.45231566 -3.38051501 -0.14254654]
 [ 1.15782128 -6.79971146  0.87144798]]

cosec matrix :
 [[-1.04283521  1.52210106  1.01075622]
 [ 2.42648664 -1.04283521  7.0861674 ]
 [-1.32134871  1.01075622  1.52210106]]

sec matrix :
 [[ 3.52532009  1.3264319  -6.87285064]
 [-1.09753791  3.52532009 -1.01010867]
 [-1.52988566 -6.87285064  1.3264319 ]]

cot matrix :
 [[-0.29581292  1.14751542 -0.14706506]
 [-2.21084541 -0.29581292 -7.01525255]
 [ 0.86369115 -0.14706506  1.14751542]]

```

## Rounding Off Values In `numpy`

- using `numpy.round`
- using `numpy.floor`
- using `numpy.ceil`
  CODE

```python
# import numpy module
import numpy as np

# create a matrix of order 3 x 3 or random values between 1.0 and 10.0
matrix = np.random.uniform(low=1.0, high=11.0, size=(3, 3))

# print the random matrix created
print(matrix, "\n\n")

# print the matrix by rounding it till 1 decimal places
print(np.round(matrix, decimals=1), "\n\n")

# print the matrix by ceiling all the values
print(np.ceil(matrix), "\n\n")

# print all the values by flooring all the values
print(np.floor(matrix), "\n\n")

```

OUTPUT

```zsh
[[ 7.6089657   9.90537042  3.50452257]
 [ 2.9751943   7.10429011  3.45634471]
 [ 8.06112758 10.82721175  3.73738542]]


[[ 7.6  9.9  3.5]
 [ 3.   7.1  3.5]
 [ 8.1 10.8  3.7]]


[[ 8. 10.  4.]
 [ 3.  8.  4.]
 [ 9. 11.  4.]]


[[ 7.  9.  3.]
 [ 2.  7.  3.]
 [ 8. 10.  3.]]


```

## Calculating The Log And Exp Of `numpy` Arrays

CODE

```python
# import numpy module
import numpy as np

# create a matrix of order 3 x 3 or random values between 1 and 10
matrix = np.random.randint(low=1, high=11, size=(3, 3))

# print the random matrix created
print(matrix, "\n\n")

# print the natural log of all the elements of the matrix
print(np.log(matrix), "\n\n")

# print the log base 10 of all the elements of the matrix
print(np.log10(matrix), "\n\n")

# print the exp of all the elements of the matrix
print(np.exp(matrix), "\n\n")
```

OUTPUT

```zsh
[[8 3 8]
 [6 1 3]
 [5 8 8]]


[[2.07944154 1.09861229 2.07944154]
 [1.79175947 0.         1.09861229]
 [1.60943791 2.07944154 2.07944154]]


[[0.90308999 0.47712125 0.90308999]
 [0.77815125 0.         0.47712125]
 [0.69897    0.90308999 0.90308999]]


[[2.98095799e+03 2.00855369e+01 2.98095799e+03]
 [4.03428793e+02 2.71828183e+00 2.00855369e+01]
 [1.48413159e+02 2.98095799e+03 2.98095799e+03]]

```

---

CODE

```python

```

OUTPUT

```zsh

```
