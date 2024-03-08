import os
import requests
import wandb

# Thay thế 'YOUR_API_KEY' bằng API key của bạn từ wandb
wandb.login(key='1eac4d04cc3cc4aed9a1409cd8eb7dc0f6537ef2')

# Thay thế 'YOUR_PROJECT_PATH' bằng path của project bạn muốn truy cập, ví dụ: 'username/projectname'
project_path = 'namkhanh2172/DomainBed2'

# Đường dẫn tới thư mục bạn muốn lưu file
# destination_folder = "./collect_wandb"
destination_folder = "./"

# Tạo thư mục nếu nó không tồn tại
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Khởi tạo API
api = wandb.Api()

# Lấy các runs từ project
runs = api.runs(path=project_path)

for run in runs:
    print(f"Processing run: {run.name}")
    # Lấy danh sách files của run
    files = run.files()
    
    # Tìm và tải xuống file results.txt
    for file in files:
        if "results.jsonl" in file.name:
            print(file.name)
            file.download(root=destination_folder, replace=True)
            file_dir = os.path.join(destination_folder, os.path.dirname(file.name))
            with open(os.path.join(file_dir, 'done'), 'w') as f:
                f.write('done')
            

