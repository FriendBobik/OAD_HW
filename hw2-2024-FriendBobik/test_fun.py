import numpy as np
from lsp import lstsq_ne
from lsp import lstsq_svd
from lsp import lstsq


A = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
b = [1, 2, 3, 4]

#print(lstsq_ne(A, b))
#print(lstsq_svd(A, b))
#print(lstsq(A, b, "ne"))
print(lstsq(A, b, "svd"))
#print(lstsq(A, b, "42"))