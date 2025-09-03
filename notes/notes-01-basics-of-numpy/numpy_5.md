# `numpy` Notes Phase - V

## Topics Covered

- Horizontal Stacking
- Vertical Stacking
- Horizontal Splitting
- Vertical Splitting

## Horizontally Stacking `numpy` Arrays

- joins any numbers of `numpy` arrays passed in the tuple in the form of a horizontal stack

CODE

```python
# import numpy module
import numpy as np

# create a matrix of 3 x 4
MatrixOne = np.arange(12).reshape(3, 4)

# create another matrix of 3 x 4
MatrixTwo = np.arange(12, 24).reshape(3, 4)

# print matrix one
print(f"matrix one : \n{MatrixOne}\n")

# print matrix two
print(f"matrix two : \n{MatrixTwo}\n")

# join both the matrices HORIZONTALLY
NewMatrix = np.hstack((MatrixOne, MatrixTwo))

# print the new matrix
print(f"new matrix : \n{NewMatrix}\n")

```

OUTPUT

```zsh
matrix one :
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

matrix two :
[[12 13 14 15]
 [16 17 18 19]
 [20 21 22 23]]

new matrix :
[[ 0  1  2  3 12 13 14 15]
 [ 4  5  6  7 16 17 18 19]
 [ 8  9 10 11 20 21 22 23]]

```

## Vertically Stacking `numpy` Arrays

- joins any numbers of `numpy` arrays passed in the tuple in the form of a vertical stack

CODE

```python
# import numpy module
import numpy as np

# create a matrix of 3 x 4
MatrixOne = np.arange(12).reshape(3, 4)

# create another matrix of 3 x 4
MatrixTwo = np.arange(12, 24).reshape(3, 4)

# print matrix one
print(f"matrix one : \n{MatrixOne}\n")

# print matrix two
print(f"matrix two : \n{MatrixTwo}\n")

# join both the matrices VERTICALLY
NewMatrix = np.vstack((MatrixOne, MatrixTwo))

# print the new matrix
print(f"new matrix : \n{NewMatrix}\n")

```

OUTPUT

```zsh
matrix one :
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]]

matrix two :
[[12 13 14 15]
 [16 17 18 19]
 [20 21 22 23]]

new matrix :
[[ 0  1  2  3]
 [ 4  5  6  7]
 [ 8  9 10 11]
 [12 13 14 15]
 [16 17 18 19]
 [20 21 22 23]]
```

---

## Horizontally Splitting `numpy` Arrays

CODE

```python
# import numpy module
import numpy as np

# create a matrix of 4 x 4
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

# print the matrix
print(f"matrix : \n{matrix}\n")

# split the matrix into two equal matrices HORIZONTALLY
MatrixOne, MatrixTwo = np.hsplit(matrix, 2)

# print both the matrices created from the matrix
print(MatrixOne, "\n\n", MatrixTwo)


```

OUTPUT

```zsh
matrix :
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]

[[ 1  2]
 [ 5  6]
 [ 9 10]
 [13 14]]

 [[ 3  4]
 [ 7  8]
 [11 12]
 [15 16]]
```

---

## Vertically Splitting `numpy` Arrays

CODE

```python
# import numpy module
import numpy as np

# create a matrix of 4 x 4
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16]])

# print the matrix
print(f"matrix : \n{matrix}\n")

# split the matrix into four equal matrices VERTICALLY
MatrixOne, MatrixTwo, MatrixThree, MatrixFour = np.vsplit(matrix, 4)

# print both the matrices created from the matrix
print(MatrixOne, "\n\n", MatrixTwo, "\n\n", MatrixThree, "\n\n", MatrixFour)


```

OUTPUT

```zsh
matrix :
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]]

[[1 2 3 4]]

 [[5 6 7 8]]

 [[ 9 10 11 12]]

 [[13 14 15 16]]
```
