import os


path1 = r'D:\database\test\clean_with_noise' #带噪
path2 = r'D:\database\test\clean' #干净

path_write = r"D:\database\test.txt"

files1 = []
files2 = []

for root1, dirs1, wavfiles1 in os.walk(path1):

    for file_single1 in wavfiles1:
        file1 = root1 + '\\' + file_single1
        # print(file1)
        files1.append(file1)


for root2, dirs2, wavfiles2 in os.walk(path2):

    for file_single2 in wavfiles2:
        file2 = root2 + '\\' + file_single2
        files2.append(file2)

# files1 = os.listdir(path1)
# files2 = os.listdir(path2)
# print(len(files1))
# print(len(files2))

with open(path_write, 'a') as txt:
     for i in range(len(files1)):
        string = os.path.join(path1, files1[i]) + ' ' + os.path.join(path2, files2[i])
        # print('string', string)
        txt.write(string + '\n')

txt.close()



