# -*- coding: UTF-8 -*-
import  chardet
import os
path = "./danzi"
#制作字典映射
dic = {}
for name in os.listdir(path):
    print  name
    dic[name] = 1
with open("./dic.txt",'w') as f:
    for i in dic:
        print  chardet.detect(i)
        print i

        f.write(i + "\n")