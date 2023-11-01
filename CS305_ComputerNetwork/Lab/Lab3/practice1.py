import os

# 获取当前工作目录
current_directory = os.getcwd()

# 获取当前目录下的所有文件和子文件夹
file_list = os.listdir(current_directory)

# 遍历文件列表
for file_name in file_list:
    # 检查是否为文件而不是文件夹
    if os.path.isfile(file_name):
        # 使用os.path.splitext分离文件名和后缀
        file_name_without_extension, file_extension = os.path.splitext(file_name)
        
        # 打印文件名和后缀
        print("文件名:", file_name_without_extension)
        print("后缀:", file_extension)
