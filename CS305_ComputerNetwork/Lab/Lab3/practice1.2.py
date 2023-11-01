import os
import mimetypes

# 获取当前工作目录
current_directory = os.getcwd()

# 获取当前目录下的所有文件
file_list = os.listdir(current_directory)

# 遍历文件列表
for file_name in file_list:
    # 拼接文件的完整路径
    file_path = os.path.join(current_directory, file_name)
    
    # 检查是否为文件而不是文件夹
    if os.path.isfile(file_path):
        # 使用mimetypes模块获取文件的MIME类型
        mime_type, _ = mimetypes.guess_type(file_path)
        
        if mime_type is not None:
            print(f"文件名: {file_name}, MIME类型: {mime_type}")
        else:
            print(f"文件名: {file_name}, 无法确定MIME类型")
