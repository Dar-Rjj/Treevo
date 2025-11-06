import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # Momentum-Volume Divergence Efficiency
    # Multi-Period Momentum Divergence
    mom_short = data['close'] / data['close'].shift(5) - 1
    mom_medium = data['close'] / data['close'].shift(10) - 1
    momentum_divergence = np.sign(mom_short * mom_medium)
    
    # Volume Confirmation Component
    vol_median = data['volume'].rolling(window=10, min_periods=5).median()
    vol_deviation = data['volume'] / vol_median
    momentum_volume = momentum_divergence * np.sqrt(np.abs(vol_deviation)) * np.sign(vol_deviation)
    
    # Efficiency Adjustment
    abs_change = np.abs(data['close'] / data['close'].shift(1) - 1)
    tr = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    efficiency_ratio = abs_change / tr.replace(0, np.nan)
    factor1 = momentum_volume * efficiency_ratio * np.sign(momentum_volume)
    factor1 = np.cbrt(factor1)
    
    # Volatility-Adaptive Gap Persistence
    # Gap Strength Measurement
    gap_pct = (data['open'] / data['close'].shift(1) - 1)
    intraday_range = data['high'] - data['low']
    gap_retention = gap_pct / intraday_range.replace(0, np.nan)
    gap_persistence = np.tanh(gap_retention * 10)
    
    # Volatility Context
    tr_atr = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    atr = tr_atr.rolling(window=10, min_periods=5).mean()
    vol_adjusted_gap = gap_persistence / np.sqrt(atr.replace(0, np.nan))
    
    # Volume-Weighted Confirmation
    vol_rank = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: (x[-1] > x).mean() if len(x) == 20 else np.nan, raw=True
    )
    factor2 = vol_adjusted_gap * vol_rank * np.sign(vol_adjusted_gap)
    
    # Intraday Pressure Efficiency Divergence
    # Opening Pressure Assessment
    opening_gap = data['open'] / data['close'].shift(1) - 1
    vol_avg_5d = data['volume'].rolling(window=5, min_periods=3).mean()
    early_vol_ratio = data['volume'] / vol_avg_5d
    
    # Closing Efficiency Measurement
    daily_range = data['high'] - data['low']
    close_change = data['close'] / data['close'].shift(1) - 1
    closing_efficiency = np.abs(close_change) / daily_range.replace(0, np.nan)
    
    # Pressure Divergence Signal
    pressure_divergence = opening_gap * closing_efficiency * np.sign(opening_gap)
    factor3 = pressure_divergence * np.cbrt(early_vol_ratio)
    
    # Range-Based Reversal Harmony
    # Extreme Move Detection
    ret_3d = data['close'] / data['close'].shift(3) - 1
    ret_20d_rank = data['close'].pct_change(periods=1).rolling(window=20, min_periods=10).apply(
        lambda x: (x[-1] > x).mean() if len(x) == 20 else np.nan, raw=True
    )
    reversal_signal = -np.sign(ret_3d) * np.where(ret_20d_rank > 0.8, 1, np.where(ret_20d_rank < 0.2, 1, 0))
    
    # Liquidity Confirmation
    vol_std_10d = data['volume'].rolling(window=10, min_periods=5).std()
    vol_spike = data['volume'] / (vol_std_10d + 1e-6)
    vol_adjusted_reversal = reversal_signal * np.sqrt(np.abs(vol_spike)) * np.sign(vol_spike)
    
    # Range Efficiency Integration
    daily_efficiency = np.abs(close_change) / tr.replace(0, np.nan)
    factor4 = vol_adjusted_reversal * daily_efficiency
    factor4 = np.tanh(factor4 * 2)
    
    # Momentum-Volatility Correlation Breakdown
    # Multi-Timeframe Momentum Correlation
    mom_5d_series = data['close'].pct_change(periods=5)
    mom_10d_series = data['close'].pct_change(periods=10)
    
    def rolling_corr(x):
        if len(x) < 5:
            return np.nan
        return np.corrcoef(x[:5], x[5:])[0, 1] if len(x) == 10 else np.nan
    
    corr_series = pd.Series(index=data.index, dtype=float)
    for i in range(9, len(data)):
        window_5d = mom_5d_series.iloc[i-4:i+1].values
        window_10d = mom_10d_series.iloc[i-4:i+1].values
        if not (np.isnan(window_5d).any() or np.isnan(window_10d).any()):
            corr_series.iloc[i] = np.corrcoef(window_5d, window_10d)[0, 1]
    
    corr_change = corr_series.diff()
    
    # Volatility Context Integration
    vol_10d = data['close'].pct_change(periods=1).rolling(window=10, min_periods=5).std()
    vol_adjusted_corr = corr_change / (vol_10d + 1e-6)
    
    # Volume Confirmation
    vol_momentum = data['volume'] / data['volume'].shift(5) - 1
    factor5 = vol_adjusted_corr * vol_momentum * np.sign(vol_adjusted_corr)
    factor5 = np.cbrt(factor5)
    
    # Combine all factors with equal weighting
    factors = pd.DataFrame({
        'f1': factor1,
        'f2': factor2,
        'f3': factor3,
        'f4': factor4,
        'f5': factor5
    })
    
    # Z-score normalization for each factor
    for col in factors.columns:
        mean_val = factors[col].rolling(window=20, min_periods=10).mean()
        std_val = factors[col].rolling(window=20, min_periods=10).std()
        factors[col] = (factors[col] - mean_val) / (std_val + 1e-6)
    
    # Equal weighted combination
    final_factor = factors.mean(axis=1)
    
    return final_factor
