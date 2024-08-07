import os

path2 = r'D:\database\test\clean'  # 干净
path1 = r'D:\database\test\clean_with_noise'  # 带噪
path_write = r"D:\database\test.txt"

# 获取文件路径和修改时间的元组列表
files1 = [(f, os.path.getmtime(os.path.join(path1, f))) for f in os.listdir(path1)]
files2 = [(f, os.path.getmtime(os.path.join(path2, f))) for f in os.listdir(path2)]

# 按照修改时间进行排序
files1.sort(key=lambda x: x[1])
files2.sort(key=lambda x: x[1])

# 只保留文件名
sorted_files1 = [f[0] for f in files1]
sorted_files2 = [f[0] for f in files2]



# 确保两个文件夹中的文件数量相同
min_length = min(len(sorted_files1), len(sorted_files2))

with open(path_write, 'a') as txt:
    for i in range(min_length):
        string = os.path.join(path1, sorted_files1[i]) + ' ' + os.path.join(path2, sorted_files2[i])
        txt.write(string + '\n')