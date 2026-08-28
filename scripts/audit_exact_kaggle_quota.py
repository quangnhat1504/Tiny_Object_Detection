import json, os, sys, datetime
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

sys.stdout.reconfigure(encoding='utf-8')

profiles = {}
for p in Path(r'C:\tmp').rglob('kaggle.json'):
    try:
        with open(p, 'r') as f:
            data = json.load(f)
            u = data.get('username')
            if u and u not in profiles:
                profiles[u] = p.parent
    except Exception:
        pass

print("| Tài Khoản (Account) | GPU Đã Dùng (Giờ) | GPU Còn Lại (Giờ) | TPU Còn Lại (Giờ) | Ngày Reset Quota |")
print("| :---                 | :---:             | :---:             | :---:             | :---:            |")

total_gpu_used = 0.0
total_gpu_left = 0.0
total_tpu_left = 0.0

for u, p in sorted(profiles.items()):
    os.environ['KAGGLE_CONFIG_DIR'] = str(p)
    api = KaggleApi()
    api.authenticate()
    q = api.quota_view()
    
    # GPU
    gpu = q.gpu_quota
    gpu_used_hrs = gpu.time_used.total_seconds() / 3600.0 if isinstance(gpu.time_used, datetime.timedelta) else 0.0
    gpu_tot_hrs = gpu.total_time_allowed.total_seconds() / 3600.0 if isinstance(gpu.total_time_allowed, datetime.timedelta) else 30.0
    gpu_rem_hrs = max(0.0, gpu_tot_hrs - gpu_used_hrs)
    
    total_gpu_used += gpu_used_hrs
    total_gpu_left += gpu_rem_hrs
    
    # TPU
    tpu = q.tpu_quota
    tpu_used_hrs = tpu.time_used.total_seconds() / 3600.0 if isinstance(tpu.time_used, datetime.timedelta) else 0.0
    tpu_tot_hrs = tpu.total_time_allowed.total_seconds() / 3600.0 if isinstance(tpu.total_time_allowed, datetime.timedelta) else 20.0
    tpu_rem_hrs = max(0.0, tpu_tot_hrs - tpu_used_hrs)
    total_tpu_left += tpu_rem_hrs
    
    reset = str(q.quota_refresh_time)[:10] if q.quota_refresh_time else "N/A"
    
    print(f"| `{u:<18}` | {gpu_used_hrs:5.2f}h / {gpu_tot_hrs:4.1f}h  | **{gpu_rem_hrs:5.2f} giờ**     | **{tpu_rem_hrs:5.2f} giờ**     | {reset}       |")

print(f"| **TỔNG CỘNG ({len(profiles)} ACCOUNTS)** | **{total_gpu_used:5.2f} giờ**  | **{total_gpu_left:5.2f} giờ**    | **{total_tpu_left:5.2f} giờ**    |                  |")
