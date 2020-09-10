import numpy as np

credit_data = np.genfromtxt('credit.txt', delimiter=',', skip_header=True)
print(credit_data)

print("Select the first row of credit_data:")
print(credit_data[0])

print("Select the fourth column of credit_data:")
print(credit_data[:, 3])

print("Select the element in row 4, column 0:")
print(credit_data[4, 0])

print("Give the distinct values of income, sorted from low to high:")
np.sort(np.unique(credit_data[:, 3]))

print("Add all the entries of the sixth column:")
np.sum(credit_data[:, 5])

print("Add the entries of each column of credit_data:")
credit_data.sum(axis=0)

print("Add the entries of each row:")
credit_data.sum(axis=1)

print("Select all rows where the first column is bigger than 27:")
print(credit_data[credit_data[:, 0] > 27])

print('Construct a vector "x" with the numbers 2, 5, 10 in that order:')
x = np.array([2, 5, 10])
print(x)

print("Construct a vector consisting of the numbers 0 through 9:")
print(np.arange(0, 10))

print("Select the *row numbers* of the rows where the first column of credit_data is bigger than 27:")
print(np.arange(0, 10)[credit_data[:, 0] > 27])

print("Draw a random sample of size 5 from the numbers 1 through 10 (without replacement):")
index = np.random.choice(np.arange(0, 10), size=5, replace=False)
print(index)

print("Select the corresponding rows:")
train = credit_data[index, ]
print(train)

print("Select all rows with row number not in 'index': "
      "(This does not delete any rows from the original credit_data.)")
test = np.delete(credit_data, index, axis=0)
print(test)

print('Consult the help page of the function "np.random.choice"')
help(np.random.choice)
