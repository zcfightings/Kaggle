from __future__ import unicode_literals

__author__ = 'zhaochen'


try:
    file = open(r"C:\Users\zhaochen\Desktop\1112.txt",'r')
    for eachline in file:
        print(eachline)
    file.close()
except IOError as e:
    print("IO Error happen:",e)

