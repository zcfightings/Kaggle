__author__ = 'zhaochen'

from functools import reduce

old_list = [1,3,5]
double = map(lambda x:x*2,old_list)
for x in double:
    print(x)
