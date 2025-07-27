import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_script(index):
    print(f"Starting run {index}")
    result = subprocess.run(['bash', './your_script.sh'], capture_output=True, text=True)
    return f"Run {index} completed:\n{result.stdout}" + (f"\nError:\n{result.stderr}" if result.stderr else '')

# 设置执行次数和最大并发数
n_runs = 5
max_workers = 3  # 最多并行几个

with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(run_script, i + 1) for i in range(n_runs)]
    
    for future in as_completed(futures):
        print(future.result())
