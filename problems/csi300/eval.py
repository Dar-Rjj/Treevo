import numpy as np
import pandas as pd
import os
import sys
import logging
import importlib.util
sys.path.insert(0, "../../../")

from scipy.stats import pearsonr
from utils.utils import get_heuristic_name


def load_heuristic_func(code_path):
    """
    动态加载临时生成的个体代码文件(如 gpt_temp_xxxx.py), 并获取启发式函数名。
    """
    spec = importlib.util.spec_from_file_location("gpt_module", code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]
    heuristic_name = get_heuristic_name(module, possible_func_names)
    heuristics = getattr(module, heuristic_name)
    return heuristics

def solve(market_data, heuristics):
    # 计算每只股票的因子值
    market_data['factor'] = market_data.groupby('stock_code').apply(lambda x: heuristics(x)).reset_index(level=0, drop=True)

    # 计算未来6日收益率
    market_data['future_return_6d'] = market_data.groupby('stock_code')['close'].shift(-6) / market_data['close'] - 1

    # 取所有日期
    start_date = pd.Timestamp('2021-01-01')
    end_date = pd.Timestamp('2024-01-01')
    all_dates = market_data.index.get_level_values('date').unique()
    all_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
    ic_values = []

    for date in all_dates:
        daily = market_data.xs(date, level='date')
        factors = daily['factor']
        returns = daily['future_return_6d']
        mask = factors.notna() & returns.notna()
        # mask = factors.notna() & returns.notna() & np.isfinite(factors) & np.isfinite(returns)
        if mask.sum() >= 2:
            ic, _ = pearsonr(factors[mask], returns[mask])
            if not np.isnan(ic):
                ic_values.append(ic)

    return np.mean(ic_values) if ic_values else 0
    

if __name__ == "__main__":
    print("[*] Running ...")
    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    code_path = sys.argv[4]
    assert mood in ['train', 'val']
    
    basepath = os.path.dirname(__file__)
    dataset_path = os.path.join(basepath, "all_stock_data.csv")

    market_data = pd.read_csv(dataset_path, parse_dates=['date'])
    market_data.set_index(['stock_code', 'date'], inplace=True)
    market_data.sort_index(inplace=True)

    heuristics = load_heuristic_func(code_path)
    mean_ic = solve(market_data, heuristics)

    os.remove(code_path)
    print("[*] Average:")
    print(abs(mean_ic))