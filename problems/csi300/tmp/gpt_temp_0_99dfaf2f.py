import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Divergence with Volume Confirmation
    ret_5d = data['close'].pct_change(5)
    ret_20d = data['close'].pct_change(20)
    mom_div = (ret_5d - ret_20d).abs()
    vol_ratio = data['volume'] / data['volume'].rolling(5).mean()
    factor1 = mom_div * vol_ratio
    
    # Volatility Regime Adjusted Return
    daily_ret = data['close'].pct_change()
    high_low_range = data['high'] - data['low']
    atr_20d = high_low_range.rolling(20).mean()
    price_range_20d = (data['high'].rolling(20).max() - data['low'].rolling(20).min())
    factor2 = (daily_ret / atr_20d) * price_range_20d
    
    # Intraday Strength Persistence
    intraday_ratio = (data['close'] - data['low']) / (data['high'] - data['low'])
    intraday_ratio_avg = intraday_ratio.rolling(5).mean()
    intraday_ratio_std = intraday_ratio.rolling(5).std()
    factor3 = intraday_ratio * (intraday_ratio_avg / intraday_ratio_std)
    
    # Volume-Price Correlation Breakout
    vol_close_corr = data['volume'].rolling(10).corr(data['close'])
    corr_avg_20d = vol_close_corr.rolling(20).mean()
    corr_deviation = vol_close_corr - corr_avg_20d
    factor4 = corr_deviation * daily_ret
    
    # Acceleration-Deceleration Indicator
    ret_5d_diff = ret_5d.diff()
    vol_5d = data['volume'].pct_change(5)
    vol_5d_diff = vol_5d.diff()
    factor5 = ret_5d_diff * vol_5d_diff * np.sign(ret_5d)
    
    # Resistance Break with Volume Surge
    resistance_20d = data['high'].rolling(20).max().shift(1)
    vol_avg_20d = data['volume'].rolling(20).mean()
    break_resistance = (data['high'] > resistance_20d).astype(int)
    vol_surge = data['volume'] / (1.5 * vol_avg_20d)
    factor6 = break_resistance * vol_surge
    
    # Mean Reversion with Volatility Scaling
    ma_10d = data['close'].rolling(10).mean()
    std_10d = data['close'].rolling(10).std()
    deviation = (data['close'] - ma_10d) / std_10d
    factor7 = deviation * np.sign(ret_5d)
    
    # Liquidity-Adjusted Momentum
    ret_10d = data['close'].pct_change(10)
    avg_trade_size = (data['amount'] / data['volume']).rolling(5).mean()
    factor8 = ret_10d / np.log(avg_trade_size)
    
    # Open-to-Close Efficiency Ratio
    intraday_ret = (data['close'] - data['open']) / data['open']
    price_range = (data['high'] - data['low']) / data['open']
    factor9 = (intraday_ret / price_range) * intraday_ret.abs()
    
    # Volume-Weighted Price Level
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    vwap = (typical_price * data['volume']).rolling(5).sum() / data['volume'].rolling(5).sum()
    vwap_deviation = (data['close'] - vwap) / vwap
    vol_slope = data['volume'].rolling(5).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0])
    factor10 = vwap_deviation * vol_slope
    
    # Combine factors with equal weights
    factors = pd.DataFrame({
        'f1': factor1, 'f2': factor2, 'f3': factor3, 'f4': factor4, 'f5': factor5,
        'f6': factor6, 'f7': factor7, 'f8': factor8, 'f9': factor9, 'f10': factor10
    })
    
    # Z-score normalization for each factor
    factors_normalized = factors.apply(lambda x: (x - x.rolling(20).mean()) / x.rolling(20).std())
    
    # Equal-weighted combination
    final_factor = factors_normalized.mean(axis=1)
    
    return final_factor
