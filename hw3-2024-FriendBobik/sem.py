# import numpy as np 
# import pandas as pd

# X=pd.read_table("/Users/aboba/Desktop/OAD/hw3-2024-FriendBobik/jla_mub.txt",
#                 sep='\s+',
#                 comment='#',
#                 header=None,
#                 names=['z','mu'])

# print (X)


from scipy.integrate import quad

z=lambda x: x**3

print (quad(z,0,1))

print (z(4))