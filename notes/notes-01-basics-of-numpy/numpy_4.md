# `numpy` Notes Phase - IV

## Topics Covered

- `numpy` Array Elements Accessing
- `numpy` Array Slicing
- Looping On `numpy` Arrays
- Flattening 2D (matrices) - 3D (tensors) `numpy` Arrays Into 1D Array (vectors)

---

## Accessing Elements Of A 1D `numpy`Array (Vector)

CODE

```python
# import numpy module
import numpy as np

# create a numpy array
vector = np.array([10, 20, 30, 40, 50])

# printing first element of the vector
print(f"first element of the vector is : {vector[0]}")

# printing the first element of the vector using -ve indexing
print(f"first element of the vector using -ve indexing is : {vector[-5]}\n")

# printing the last element of the vector
print(f"last element of the vector is  : {vector[4]}")

# printing the last element of the vector using -ve indexing
print(f"last element of the vector using -ve indexing : {vector[-1]}\n")

# printing the middle element element (= 30) using indexing
print(f"middle element : {vector[2]}")

# printing the middle element (= 30) using -ve indexing
print(f"middle element using -ve indexing : {vector[-3]}")

```

OUTPUT

```zsh
first element of the vector is : 10
first element of the vector using -ve indexing is : 10

last element of the vector is : 50
last element of the vector using -ve indexing : 50

middle element : 30
middle element using -ve indexing : 30
```

---

## Accessing Elements Of A 2D `numpy` Array (Matrix)

CODE

```python
# import numpy module
import numpy as np

# create a matrix
matrix = np.array([[10, 20, 30, 40, 50],
                   [60, 70, 80, 90, 100],
                   [110, 120, 130, 140, 150],
                   [160, 170, 180, 190, 200],
                   [210, 220, 230, 240, 250]])

# print the first element of the matrix
print(f"first element of the matrix is : {matrix[0][0]} or {matrix[0, 0]}")

# print the last element of the matrix
print(f"last element of the matrix is : {matrix[4][4]} or {matrix[4, 4]}\n")

# printing 100 from the matrix
print(matrix[1, 4])

# printing 240 from the matrix
print(matrix[4, 3])


```

OUTPUT

```zsh
first element of the matrix is : 10 or 10
last element of the matrix is : 250 or 250

100
240
```

---

## Accessing Elements Of A 3D `numpy` Array (Tensor)

CODE

```python
# import numpy module
import numpy as np

# create a tensor
tensor = np.array([[[1, 2],
                    [3, 4]],
                   [[5, 6],
                    [7, 8]]])

# print first element of the tensor
print(f"first element of the tensor is : {tensor[0, 0, 0]}")

# print the last element of the tensor
print(f"last element of the tensor is : {tensor[1, 1, 1]}\n")

# printing 4 from the tensor
print(tensor[0, 1, 1])

# printing 5 from the tensor
print(tensor[1, 0, 0])

```

OUTPUT

```zsh
first element of the tensor is : 1
last element of the tensor is : 8

4
5
```

---

## Slicing A 1D `numpy` Array (Vector)

CODE

```python
# import numpy module
import numpy as np

# create a vector
vector = np.array([10, 20, 30, 40, 50, 60, 70, 80])

# print the elements of the array from 10 to 40
print(f"elements of array from 10 to 40 : {vector[0 : 4]}")

# print the elements of the array from 60 and 70
print(f"elements of array from 60 to 70 : {vector[5 : 7]}")

# print the elements of the array from 50 to the end of the array
print(f"elements from 50 to the end of the array : {vector[4 : ]}")

# print the array in reverse order
print(f"elements of the array in reverse order : {vector[ : : -1]}")

# print the elements of the array from 40 to 70 using -ve slicing
print(f"elements from 40 to 70 using -ve slicing : {vector[-5 : -1]}")

# print the elements of the array from 40 to 80 using -ve slicing
print(f"elements from 40 to 80 using -ve slicing : {vector[-5 : ]}")
```

OUTPUT

```zsh
elements of array from 10 to 40 : [10 20 30 40]
elements of array from 60 to 70 : [60 70]
elements from 50 to the end of the array : [50 60 70 80]
elements of the array in reverse order : [80 70 60 50 40 30 20 10]
elements from 40 to 70 using -ve slicing : [40 50 60 70]
elements from 40 to 80 using -ve slicing : [40 50 60 70 80]
```

---

## Slicing A 2D `numpy` Array (Matrix)

> syntax : `<matrix_name>[row_start : row_stop : row_step, col_start : col_stop : col_step]`

CODE

```python
# import numpy module
import numpy as np

# create a matrix
matrix = np.array([[1, 2, 3, 4, 5],
                   [6, 7, 8, 9, 10],
                   [1, 2, 3, 4, 15],
                   [0, 7, 8, 9, 20]])

# print the first row all columns
print(f"first row, all columns : {matrix[0, : ]}")

# print the second column, all rows
print(f"print the second column, all rows : {matrix[ :, 1]}")

# print last row but only till 3 columns
print(f"last rows but till 3 columns {matrix[3, : 3]}")

# print the last column but only till last two rows
print(f"last column with only last two values : {matrix[2:, 4]}")

# print the first 2 values of the first column
print(f"first two values of the first column : {matrix[0 : 2 , 0]}")

# IMPORTANT
# print 1, 5, 0, 20
print(f"1, 5, 0, 20 from the matrix : \n{matrix[::3, ::4]}")
```

OUTPUT

```zsh
first row, all columns : [1 2 3 4 5]
print the second column, all rows : [2 7 2 7]
last rows but till 3 columns [0 7 8]
last column with only last two values : [15 20]
first two values of the first column : [1 6]
1, 5, 0, 20 from the matrix :
[[ 1  5]
 [ 0 20]]
```

---

## Slicing A 3D `numpy` Array (Tensor)

> syntax : `<matrix_name>[tensor_start : tensor_stop : tensor_step, row_start : row_stop : row_step, col_start : col_stop : col_step]`

CODE

```python
# import numpy module
import numpy as np

# create a tensor of 3 x 3 x 3
tensor = np.array([[[0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8]],

                   [[9, 10, 11],
                    [12, 13, 14],
                    [15, 16, 17]],

                   [[18, 19, 20],
                    [21, 22, 23],
                    [24, 25, 26]]])

# print the tensor
print(f"tensor : \n{tensor}\n")

# print the second matrix of the tensor
print(f"second matrix of the tensor is : \n{tensor[1]}\n")

# print the first and the last matrix of the tensor
print(f"first and last matrix of the tensor are : \n{tensor[::2]}")

# print the second row of the first matrix
print(f"second row of the first matrix is : \n{tensor[0, 1,]}\n")

# print the middle column of the second matrix
print(f"the middle column of the second matrix is : \n{tensor[1, :, 1]}\n")

# IMPORTANT
# print 22, 23, 25, 26
print(f"22, 23, 25, 26 from the tensor : \n{tensor[2, 1:, 1:]}\n")

# IMPORTANT
# print 0, 2, 18, 20
print(f"0, 2, 18, 20 from the tensor : \n{tensor[::2, 0, ::2]}\n")
```

OUTPUT

```zsh
tensor :
[[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]]

 [[ 9 10 11]
  [12 13 14]
  [15 16 17]]

 [[18 19 20]
  [21 22 23]
  [24 25 26]]]

second matrix of the tensor is :
[[ 9 10 11]
 [12 13 14]
 [15 16 17]]

first and last matrix of the tensor are :
[[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]]

 [[18 19 20]
  [21 22 23]
  [24 25 26]]]
second row of the first matrix is :
[3 4 5]

the middle column of the second matrix is :
[10 13 16]

22, 23, 25, 26 from the tensor :
[[22 23]
 [25 26]]

0, 2, 18, 20 from the tensor :
[[ 0  2]
 [18 20]]
```

---

## Looping On 1D `numpy` Arrays (Vectors)

CODE

```python
# import numpy module
import numpy as np

# create a vector of 5 elements
vector = np.array([10, 20, 30, 40, 50])

# print the vector using pythonic for-each loop
for i in vector:
    print(i)

# for readability
print()

# print the vector using range-based loop
for i in range(vector.size):
    print(vector[i])
```

OUTPUT

```zsh
10
20
30
40
50

10
20
30
40
50
```

---

## Looping On 2D `numpy` Arrays (Matrices)

CODE

```python
# import numpy module
import numpy as np

# create a vector of 5 elements
matrix = np.array([[0, 1, 2],
                  [3, 4, 5],
                  [6, 7, 8]])

# print the matrix
print(matrix, "\n")

# print the matrix row by row using pythonic for-each loop
for i in matrix:
    print(i)

# for readability
print()

# print the matrix element by element using iterator
for i in np.nditer(matrix):
    print(i)
```

OUTPUT

```zsh
[[0 1 2]
 [3 4 5]
 [6 7 8]]

[0 1 2]
[3 4 5]
[6 7 8]

0
1
2
3
4
5
6
7
8
```

---

## Looping On 3D `numpy` Arrays (Tensors)

CODE

```python
# import numpy module
import numpy as np

# create a tensor of 3 x 3 x 3
tensor = np.array([[[0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8]],

                   [[9, 10, 11],
                    [12, 13, 14],
                    [15, 16, 17]],

                   [[18, 19, 20],
                    [21, 22, 23],
                    [24, 25, 26]]])

# print the tensor
print(tensor, "\n")

# print each matrix in the tensor by using pythonic for-each loop
for i in tensor:
    print(i)

# for readability
print("\n")

# print the tensor element by element by using iterator
for i in np.nditer(tensor):
    print(i)

```

OUTPUT

```zsh
[[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]]

 [[ 9 10 11]
  [12 13 14]
  [15 16 17]]

 [[18 19 20]
  [21 22 23]
  [24 25 26]]]

[[0 1 2]
 [3 4 5]
 [6 7 8]]
[[ 9 10 11]
 [12 13 14]
 [15 16 17]]
[[18 19 20]
 [21 22 23]
 [24 25 26]]


0
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
```

---

## Flattening 2D (matrices) - 3D (tensors) `numpy` Arrays Into 1D `numpy` Arrays (vectors)

CODE

```python
# import numpy module
import numpy as np

# create a matrix of 3 x 3
matrix = np.array([[0, 1, 2],
                   [3, 4, 5],
                   [6, 7, 8]])

# create a tensor of 3 x 3 x 3
tensor = np.array([[[0, 1, 2],
                    [3, 4, 5],
                    [6, 7, 8]],

                   [[9, 10, 11],
                    [12, 13, 14],
                    [15, 16, 17]],

                   [[18, 19, 20],
                    [21, 22, 23],
                    [24, 25, 26]]])

# print the matrix
print(matrix, "\n")

# print the tensor
print(tensor, "\n")

# convert the matrix into a vector
converted_matrix = np.ravel(matrix)

# print the converted matrix
print(converted_matrix, "\n")

# convert the tensor into a vector
converted_tensor = np.ravel(tensor)

# print the converted tensor
print(converted_tensor)


```

OUTPUT

```zsh
[[0 1 2]
 [3 4 5]
 [6 7 8]]

[[[ 0  1  2]
  [ 3  4  5]
  [ 6  7  8]]

 [[ 9 10 11]
  [12 13 14]
  [15 16 17]]

 [[18 19 20]
  [21 22 23]
  [24 25 26]]]

[0 1 2 3 4 5 6 7 8]

[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26]
```

---
