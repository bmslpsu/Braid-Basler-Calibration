import zipfile
import os
import gzip
import shutil

file_path = "./data_file_4_angled/20250408_215614.braidz"
extract_path = "./data_file_4_angled/wand_data"

file_path = f'J:/barid_cal/20250416_175127.braidz'
extract_path = f'J:/barid_cal/20250416_175127'

# 创建解压目录（如果不存在）
os.makedirs(extract_path, exist_ok=True)

# 解压 .braidz 文件
with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("解压 .braidz 完成！")

# 找到 .gz 文件并解压
gz_file_path = os.path.join(extract_path, "data2d_distorted.csv.gz")
csv_file_path = os.path.join(extract_path, "data2d_distorted.csv")

if os.path.exists(gz_file_path):
    with gzip.open(gz_file_path, 'rb') as gz_file, open(csv_file_path, 'wb') as csv_file:
        shutil.copyfileobj(gz_file, csv_file)
    print(f"解压 {gz_file_path} 完成！")
else:
    print(f"未找到 {gz_file_path}，请检查文件路径！")

# 找到 .gz 文件并解压
gz_file_path = os.path.join(extract_path, "cam_info.csv.gz")
csv_file_path = os.path.join(extract_path, "cam_info.csv")

if os.path.exists(gz_file_path):
    with gzip.open(gz_file_path, 'rb') as gz_file, open(csv_file_path, 'wb') as csv_file:
        shutil.copyfileobj(gz_file, csv_file)
    print(f"解压 {gz_file_path} 完成！")
else:
    print(f"未找到 {gz_file_path}，请检查文件路径！")