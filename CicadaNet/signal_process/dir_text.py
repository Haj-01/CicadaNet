#!/usr/bin/python
import os

txtName = r"E:\HAN\database\test.txt"
f = open(txtName, "a+")
dire = r'E:\HAN\database\test\clean_with_noise'



for root, dirs, files in os.walk(dire):

    for file_single in files:
        test = root + '\\' + file_single
        # print(test)
        refile = file_single[0:8]
        result = test + '\n'
        f.write(result)
f.close()


# file_namelist = os.listdir(dire)
# filename = str(file_namelist)
# # f = open(dire + "\\" + "DF1.txt", "a+")
# f.write(filename)
# files = os.listdir(dire)
# files.sort(key=lambda x:int(x[:-4]))#倒着数第四位'.'为分界线，按照‘.’左边的数字从小到大排序
# l = len(files)
#
# for i in range(0, l):
#     test = dire + '\\' + files[i]
#     result = test + '\n'
#     f.write(result)
# f.close()




# import os
#
#
# def ListFilesToTxt(dir, file, wildcard, recursion):
#     exts = wildcard.split(" ")
#     files = os.listdir(dir)
#     for name in files:
#         fullname = os.path.join(dir, name)
#         if (os.path.isdir(fullname) & recursion):
#             ListFilesToTxt(fullname, file, wildcard, recursion)
#         else:
#             for ext in exts:
#                 if (name.endswith(ext)):
#                     file.write(name + "\n")
#                     break
#
#
# def Test():
#     dir = r'D:\birddata\B003\segment'  # 文件路径
#     outfile = "trainaudio.txt"  # 写入的txt文件名
#     wildcard = ".wav"  # 要读取的文件类型；
#
#     file = open(outfile, "w")
#     if not file:
#         print("cannot open the file %s for writing" % outfile)
#
#     ListFilesToTxt(dir, file, wildcard, 1)
#
#     file.close()
#
# Test()
