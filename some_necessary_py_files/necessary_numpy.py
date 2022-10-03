import sys
import numpy as np

# 1D array => shape: (4,)
# 2D array => shape: (2,3) 2x3 matrix => axis 0 is for the rows, axis 1 is for the columns
# 3D array => shpae: (4, 3, 2) 4x3x2 matrix

# NumPy is faster than Lists => less bytes of memory , no type checking, 
# a = [1,3,5] => a = np.array([1,3,5])

a = np.array([1,2,3])
b = np.array([[9.0,8.0,7.0],[1.0,2.0,3.0]])
print(a)
print(a.ndim)
print(a.shape)

print(b)
print(b.ndim)
print(b.shape)

print(b[0,1])

# Get type
print(b.dtype)
a = np.array([1,2,3],dtype="int32")

# Get size
print(a.itemsize)

############################################################
# ACCESSING/CHANGING SPECIFIC ELEMENTS, ROWS, COLUMNS, ETC #
############################################################

a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
print(a)
print(a.shape)

# get a specific element [r ,c]
print(a[1,5]) # will give us 13

# get a specific row
print(a[0,:])

# get a specific col
print(a[:,0])

# getting little more fancy [startindex:endindex:stepsize]
print(a[0,1:6:2])
a[1,5] = 20
a[:,2] = [1,2]

# 3d example

b = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(b)
print(b.shape)
print(b[0,1,1]) # to get 4

# replace
b[:,1,:]  = [[9,9],[8,8]] 

#########################################
# INITIALIZE DIFFERENT TYPES OF ARRAYS  #
#########################################

# all 0s
a = np.zeros((2,3))

# all 1s 
b = np.ones((3,2),dtype="int32")

# any other number
np.full((2,2), fill_value=99, dtype="float32")

# any other number (full_like)
a = np.array([[1,2,3,4,5,6,7],[8,9,10,11,12,13,14]])
b = np.full(a.shape,4)

# random decimal numbers
b = np.random.rand(4,2)
b = np.random.random_sample(a.shape)

# random integer values
b = np.random.randint(4,7, size=(3,3))

# identity matrix
b = np.identity(3)

# repeat the array
arr = np.array([[1,2,3]])
r1 = np.repeat(arr,3,axis=0) # repeat the array three times

# to create a specific matrix
output = np.ones((5,5))
z = np.zeros((3,3))
z[1,1] = 9
output[1:-1,1:-1] = z

# be careful when copying arrays!
a = np.array([1,2,3])
b = a
b[0] = 100
print(a) # a's first element also becomes 100

# to prevent this we should do the copying via
b = a.copy()


###############
# MATHEMATICS #
###############

a = np.array([1,2,3,4])
a +=2
a -=2
a *=2
a /=2

b = np.array([1,0,1,0])
c = a+b
a **= 2

# take sine of all values
np.sin(a)

##################
# LINEAR ALGEBRA #
##################

# lets say we have two matrices
a = np.ones((2,3))
b = np.full((3,2),2)

# matrix multiplication
c = np.matmul(a,b)

# find the determinant
c = np.identity(3)
d = np.linalg.det(c)


##############
# STATISTICS #
##############

stats = np.array([[1,2,3],[4,5,6]])

np.min(stats,axis=1)
np.max(stats,axis=0)
np.sum(stats,axis=0)

#######################
# REORGANIZING ARRAYS #
#######################

before = np.array([[1,2,3,4],[4,6,7,8]])
print(before)

after = before.reshape((8,1))
after2 = before.reshape((4,2))

# Vertically stacking vectors

v1 = np.array([1,2,3,4])
v2 = np.array([5,6,7,8]) 

v3 = np.vstack([v1,v2])

# Horizontally stacking vectors

h1 = np.ones((2,4))
h2 = np.zeros((2,2))

h3 = np.hstack((h1,h2))

#################
# MISCELLANEOUS #
#################

# load data from file 
filedata = np.genfromtxt('data.txt', delimiter=',')
filedata = filedata.astpye('int32')

# Boolean Masking and Advanced Indexing

boolean_filedata = filedata > 50
values_bigger_than_fifthy = filedata[filedata > 50]

# you can index with a list in numpy
a = np.array([1,2,3,4,5,6,7,8,9])
b = a[[1,2,5]]

isit = np.any(filedata>50, axis=0)
isit = np.all(filedata>50, axis=1)

values_bigger_than_fifthy_smaller_than_hudnred = ((filedata > 50) & (filedata < 100))
values_not_bigger_than_fifthy_smaller_than_hudnred = (~((filedata > 50) & (filedata < 100)))