# `numpy` Notes Phase - I

## What Is `numpy` ?

(pronounced /ˈnʌmpaɪ/ NUM-py) is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays. The predecessor of NumPy, Numeric, was originally created by Jim Hugunin with contributions from several other developers. In 2005, Travis Oliphant created NumPy by incorporating features of the competing Numarray into Numeric, with extensive modifications. NumPy is open-source software and has many contributors. NumPy is fiscally sponsored by NumFOCUS.

[learn more about numpy](https://en.wikipedia.org/wiki/NumPy)

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

## Why To Use NumPy ? (Time Efficiency Of Using `numpy` Arrays over Python List)

CODE :

```python
# import numpy module
import numpy as np

# import time module to measure time taken by operations
import time

# ===== Pythonic List Section =====

# note the start time of the operation
start_time = time.time()

# create a pythonic list using list comprehension
ListOne = [i for i in range(100000000)]

# create another list using list comprehension
ListTwo = [i for i in range(100000000, 200000000)]

# create another list which with hold the sum of ListOne[i] and ListTwo[i]
SumList = []

# add ListOne[i] and ListTwo[i] and append it in SumList
for i in range(len(ListOne)):
    SumList.append(ListOne[i] + ListTwo[i])

# print the time taken by the operation
print(f"time taken by pythonic lists : {time.time() - start_time}\n")

# ===== Numpy Array Section =====

# note the start time of the operation
start_time = time.time()

# creating numpy array
NumpyArrayOne = np.arange(100000000)

# creating another numpy array
NumpyArrayTwo = np.arange(100000000, 200000000)

# sum of both the arrays
NumpySumArray = NumpyArrayOne + NumpyArrayTwo

# print the time taken by the operation
print(f"time taken by numpy array : {time.time() - start_time}\n")

```

OUTPUT

```zsh
time taken by pythonic lists : 27.521579027175903

time taken by numpy array : 1.1130650043487549
```

---

## Why To Use NumPy ? ((Time Efficiency Of Using `numpy` Arrays over Python List))

CODE

```python
# import numpy module
import numpy as np

# import the sys module to find the memory of objects
import sys

# ===== Pythonic List Section ===== #

# create a pythonic list using list comprehension
PythonicList = [i for i in range(100000000)]

# print the memory used by the pythonic list
print(f"memory consumed by pythonic list : {sys.getsizeof(PythonicList)}")

# ===== Numpy Array Section ===== #

# create a numpy array
NumpyArray = np.arange(100000000, dtype=np.int32)

# print the memory used by the numpy array
print(f"memory consumed by pythonic list : {sys.getsizeof(NumpyArray)}")


```

OUTPUT

```zsh
memory consumed by pythonic list : 835128600
memory consumed by pythonic list : 400000112
```

---
