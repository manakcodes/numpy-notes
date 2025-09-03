# `numpy` Notes Phase - VII

## Topics Covered

- some useful functions in `numpy`

### `numpy.sort(numpy_array)` On 1D `numpy` Array (Vector)

CODE

```python
# import numpy module
import numpy as np

# create a random vector
vector = np.random.randint(low=1, high=11, size=(10,))

# print the original vector
print(f"vector : \n{vector}\n")

# print the vector sorted in ascending order
print(f"vector sorted in ascending order is : \n{np.sort(vector)}\n")

# print the vector sorted in descending order
print(f"vector sorted in descending order is : \n{np.sort(vector)[::-1]}\n")
```

OUTPUT

```zsh
vector :
[ 2  8  6 10  1  8 10  5  2  2]

vector sorted in ascending order is :
[ 1  2  2  2  5  6  8  8 10 10]

vector sorted in ascending order is :
[10 10  8  8  6  5  2  2  2  1]
```

### `numpy.sort(numpy_array)` On 2D `numpy` Array (Matrix)

CODE

```python
# import numpy module
import numpy as np

# create a random matrix
matrix = np.random.randint(low=1, high=11, size=(3, 3))

# print the original matrix
print(f"matrix : \n{matrix}\n")

# print the matrix sorted row wise
print(f"row wise sorted matrix : \n{np.sort(matrix, axis=1)}\n")

# print the matrix sorted column wise
print(f"column wise sorted matrix : \n{np.sort(matrix, axis=0)}\n")

```

OUTPUT

```zsh
matrix : 
[[ 6 10  7]
 [ 3  3 10]
 [ 3  6  1]]

row wise sorted matrix : 
[[ 6  7 10]
 [ 3  3 10]
 [ 1  3  6]]

column wise sorted matrix : 
[[ 3  3  1]
 [ 3  6  7]
 [ 6 10 10]]

```

---

CODE

```python

```

OUTPUT

```zsh

```
