# `numpy` Notes Phase - VI

## Topics Covered

- Fancy Indexing
- Boolean Indexing

---

## Fancy Indexing

### Fancy Indexing On `numpy` 1D Arrays (Vectors)

CODE

```python
# import numpy module
import numpy as np

# create a vector
vector = np.array([10, 20, 30, 40, 50])

# print the vector
print(f"vector : \n{vector}\n")

# print the first, third and last element of the vector
print(f"first, third and last element from the vector : \n{vector[[0, 2, -1]]}\n")
```

OUTPUT

```zsh
vector :
[10 20 30 40 50]

first, third and last element from the vector :
[10 30 50]
```

---

### Fancy Indexing On `numpy` 2D Arrays (Matrices)

CODE

```python
# import numpy module
import numpy as np

# create a matrix
matrix = np.array([[1, 2, 3, 4],
                   [5, 6, 7, 8],
                   [9, 10, 11, 12],
                   [13, 14, 15, 16],
                   [17, 18, 19, 20],
                   [21, 22, 23, 24]])

# print the matrix
print(matrix, "\n\n")

# print the first, third and last row from the matrix
# hint : use fancy indexing
print(f"first, third and last row from the matrix : \n{matrix[[0, 2, 5]]}\n")

# print the first, third and last column from the matrix
# hint : use fancy indexing
print(f"first, third and last column from the matrix : \n{matrix[:, [0, 2, 3]]}\n")
```

OUTPUT

```zsh
[[ 1  2  3  4]
 [ 5  6  7  8]
 [ 9 10 11 12]
 [13 14 15 16]
 [17 18 19 20]
 [21 22 23 24]]


first, third and last row from the matrix :
[[ 1  2  3  4]
 [ 9 10 11 12]
 [21 22 23 24]]

first, third and last column from the matrix :
[[ 1  3  4]
 [ 5  7  8]
 [ 9 11 12]
 [13 15 16]
 [17 19 20]
 [21 23 24]]

```

### Fancy Indexing On `numpy` 3D Arrays (Tensors)

CODE

```python
# import numpy module
import numpy as np

# create a tensor
tensor = np.array([[[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]],

                   [[10, 11, 12],
                    [13, 14, 15],
                    [16, 17, 18]],

                   [[19, 20, 21],
                    [22, 23, 24],
                    [25, 26, 27]]])

# print the tensor
print(f"tensor : \n{tensor}\n")

# print the first and last element of the tensor
print(f"first, third and last element from the vector : \n{tensor[[0, 2]]}\n")
```

OUTPUT

```zsh
tensor :
[[[ 1  2  3]
  [ 4  5  6]
  [ 7  8  9]]

 [[10 11 12]
  [13 14 15]
  [16 17 18]]

 [[19 20 21]
  [22 23 24]
  [25 26 27]]]

first, third and last element from the vector :
[[[ 1  2  3]
  [ 4  5  6]
  [ 7  8  9]]

 [[19 20 21]
  [22 23 24]
  [25 26 27]]]

```

## Boolean Indexing

### Boolean Indexing On `numpy` 1D Arrays (Vectors)

CODE

```python
# import numpy module
import numpy as np

# create a random vector
vector = np.random.randint(low=40, high=60, size=(5,))

# print the vector
print(f"vector : \n{vector}\n")

# print all the elements which are greater than 50
print(f"elements greater than 50 are : \n{vector[vector > 50]}\n")

# print all the elements equal to 50
print(f"elements equal to 50 : \n{vector[vector == 50]}\n")

# print all the elements which are less than 50
print(f"elements less than 50 are : \n{vector[vector < 50]}\n")

```

OUTPUT

```zsh
vector :
[56 50 49 55 55]

elements greater than 50 are :
[56 55 55]

elements equal to 50 :
[50]

elements less than 50 are :
[49]

```

---

### Boolean Indexing On `numpy` 2D Arrays (Matrices)

CODE

```python
# import numpy module
import numpy as np

# create a random matrix
matrix = np.random.randint(low=1, high=3, size=(5, 5))

# print the matrix
print(f"vector : \n{matrix}\n")

# print all the matrix elements which are even
print(f"even elements : \n{matrix[matrix % 2 == 0]}\n")

# print all the matrix elements which are odd
print(f"odd elements : \n{matrix[matrix % 2 != 0]}\n")


```

OUTPUT

```zsh
vector :
[[1 2 1 1 2]
 [1 2 1 1 2]
 [1 1 1 2 2]
 [1 1 2 2 1]
 [2 1 2 2 2]]

even elements :
[2 2 2 2 2 2 2 2 2 2 2 2]

odd elements :
[1 1 1 1 1 1 1 1 1 1 1 1 1]

```

### Boolean Indexing On `numpy` 3D Arrays (Tensors)

CODE

```python
# import numpy module
import numpy as np

# create a random tensor
tensor = np.random.randint(low=1, high=100, size=(2, 2, 2))

# print the tensor
print(f"tensor : \n{tensor}\n")

# print all the elements which are divisible by 3 and 5 from the tensor
print(f"elements divisible by 3 and 5 from the tensor are : \n{tensor[(tensor % 3 == 0) & (tensor % 5 == 0)]}\n")
```

OUTPUT

```zsh
tensor :
[[[15 63]
  [43 96]]

 [[18 65]
  [86 21]]]

elements divisible by 3 and 5 from the tensor are :
[15]

```

---
