import os

txtName = r"G:\JNT\code\crnn\lifan\testset\sn10\noisy.txt"
# f = open(txtName, "a+",encoding='utf-8')
f = open(txtName, "a+")

dire = r"G:\JNT\code\crnn\lifan\testset\sn10\noisy"

for root, dirs, files in os.walk(dire):
    for file_single in files:
        test = root + '\\' +file_single
        refile = file_single[0:8]
        result = test +'\n'
        f.write(result)
f.close()