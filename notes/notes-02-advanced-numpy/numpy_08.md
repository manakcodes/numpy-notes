# `numpy` Notes Phase - VII

## Topics Covered

- Defining User Defined Functions On `numpy` Arrays
- Missing Values In `numpy` Arrays

### Defining User Defined Function On `numpy` Arrays

- Sigmoid Function  
  S(x) = 1 / (1 + e \*\* (-x))

CODE

```python
# import numpy module
import numpy as np

# create a method to find the sigmoid
def sigmoid(array):
    return ((1) / (1 + np.exp(-array)))


# create a vector
vector = np.arange(10)

# print the sigmoid of the vector
print(f"sigmoid of numpy array is : \n{sigmoid(vector)}\n")
```

OUTPUT

```zsh
sigmoid of numpy array is :
[0.5        0.73105858 0.88079708 0.95257413 0.98201379 0.99330715
 0.99752738 0.99908895 0.99966465 0.99987661]

```

- Mean Squared Error Of linear Regression

MSE = (1 / n) sigma (Y(actual) - Y(predicted)) \*\* 2

CODE

```python
# import numpy module
import numpy as np

# create a method to find the mean squared error
def mean_squared_error(actual, predicted):
    return np.mean((actual - predicted) ** 2)


# create a random vector for actual values
actual_values = np.random.randint(low=1, high=101, size=(10,))

# create a random vector for predicted values
predicted_values = np.random.randint(low=1, high=101, size=(10,))

# print the actual and predicted values vector
print(actual_values, "\n\n", predicted_values)

# print the mean squared error
print(f"mean squared error is : {mean_squared_error(actual_values, predicted_values)}\n")
```

OUTPUT

```zsh
[47 38  8 42 51 46  4 23 26 27]

 [57 58 65 34 59 47 11 74 32 66]
mean squared error is : 808.5
```

---

### Missing Values In `numpy` Arrays

```python
# import numpy module
import numpy as np

# create a vector with some missing values
vector = np.array([10, 20, 30, np.nan, 50, 60, 70, np.nan])

# print the vector
print(f"vector : \n{vector}\n")

# create a cleaned vector without missing values
cleaned_vector = vector[~ np.isnan(vector)]

# print the cleaned vector
print(f"cleaned vector : \n{cleaned_vector}\n")
```

OUTPUT

```zsh
vector :
[10. 20. 30. nan 50. 60. 70. nan]

cleaned vector :
[10. 20. 30. 50. 60. 70.]
```

---
