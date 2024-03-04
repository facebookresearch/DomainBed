import os
import requests
import wandb

# Thay thế 'YOUR_API_KEY' bằng API key của bạn từ wandb
wandb.login(key='1eac4d04cc3cc4aed9a1409cd8eb7dc0f6537ef2')

# Thay thế 'YOUR_PROJECT_PATH' bằng path của project bạn muốn truy cập, ví dụ: 'username/projectname'
project_path = 'scalemind/DomainBed'

# Khởi tạo API
api = wandb.Api()

# Lấy các runs từ project
runs = api.runs(path=project_path)

for run in runs:
    # Tạo folder cho mỗi run, sử dụng tên của run
    run_folder = run.name
    os.makedirs(run_folder, exist_ok=True)
    
    # Lấy thông tin files từ run
    files = run.files()
    
    for file in files:
        # Tạo URL download và tên file đích
        download_url = file.url
        destination_path = os.path.join(run_folder, file.name)
        
        # Tải file và lưu vào folder tương ứng
        response = requests.get(download_url, stream=True)
        if response.status_code == 200:
            with open(destination_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded {file.name} to {destination_path}")
        else:
            print(f"Failed to download {file.name}")

print("All files downloaded.")
