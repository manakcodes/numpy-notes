# `numpy` Notes Phase - III

## Topics Covered

- Vector Operations On `numpy` Arrays
- Scalar Operations `numpy` Arrays
- Relational Checking On `numpy` Matrices
- `numpy` Statistical Functions
- Row - Wise Operations In `numpy`
- Column - Wise Operations In `numpy`
- Trigonometric Operations In `numpy`
- Rounding Off Values In `numpy`
- Log And Exp Values In `numpy`

---

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

---

## Scalar Operations On `numpy` Matrices

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

# print the matrix by adding a scalar k = 2 to it
print(MatrixOne + 2, "\n\n")

# print the transpose of the matrix
print(np.transpose(MatrixOne))



```

OUTPUT

```zsh
[[1 1 1]
 [2 2 2]
 [3 3 3]]


[[3 3 3]
 [4 4 4]
 [5 5 5]]


[[1 2 3]
 [1 2 3]
 [1 2 3]]

```

---

## Relational Checking On `numpy` Matrices

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

---

## Statistical Functions In `numpy`

- `numpy.min`, `numpy.max`, `numpy.sum`, `numpy.mean`, `numpy.mode`, `numpy.median`, `numpy.std`, `numpy.var`, `numpy.cumsum`

- for row wise operations set _`axis=1`_ (row wise : left to right)
- for column wise operations set _`axis=0`_ (column wise : top to bottom)
- the default value of _`axis`_ is 0 (column wise)

---

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

---

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

---

## Using `numpy` Trigonometric Functions

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

---

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

---

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
