import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def run_script(index):
    print(f"Starting run {index}")
    # result = subprocess.run(['bash', './test_csi300.sh'], capture_output=True, text=True)
    result = subprocess.run(['bash', './test_csi500.sh'], capture_output=True, text=True)
    # result = subprocess.run(['bash', './test_dji.sh'], capture_output=True, text=True)
    # result = subprocess.run(['bash', './test_spx.sh'], capture_output=True, text=True)
    # result = subprocess.run(['bash', './test_ndx.sh'], capture_output=True, text=True)
    return f"Run {index} completed:\n{result.stdout}" + (f"\nError:\n{result.stderr}" if result.stderr else '')

# 设置执行次数和最大并发数
n_runs = 3
max_workers = 10  # 最多并行几个

futures = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    for i in range(n_runs):
        futures.append(executor.submit(run_script, i + 1))
        time.sleep(2)
    
    for future in as_completed(futures):
        print(future.result())
