import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Divergence Momentum
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Short-term momentum components
    mom_5d = close / close.shift(5) - 1
    mom_10d = close / close.shift(10) - 1
    mom_ratio = (close / close.shift(5)) / (close.shift(5) / close.shift(10))
    
    # Volume confirmation signals
    vol_trend = volume / volume.shift(5)
    vol_mom = (volume / volume.shift(5)) / (volume.shift(5) / volume.shift(10))
    
    # Volume persistence (count of days with volume > previous day's volume over last 5 days)
    vol_persistence = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        window = volume.iloc[i-5:i+1]
        vol_persistence.iloc[i] = (window > window.shift(1)).iloc[1:].sum()
    
    # Divergence detection
    bullish_div = (mom_5d > mom_5d.rolling(10).mean()) & (vol_trend < 1)
    bearish_div = (mom_5d < mom_5d.rolling(10).mean()) & (vol_trend > 1)
    div_strength = mom_5d * (1 / vol_trend)
    
    # Range Efficiency Momentum
    daily_eff = abs(close - close.shift(1)) / (high - low)
    
    # 3-day efficiency
    eff_3d = pd.Series(index=df.index, dtype=float)
    for i in range(3, len(df)):
        price_change = abs(close.iloc[i] - close.iloc[i-3])
        range_sum = sum(high.iloc[j] - low.iloc[j] for j in range(i-2, i+1))
        eff_3d.iloc[i] = price_change / range_sum if range_sum > 0 else 0
    
    # Gap-adjusted efficiency
    gap_adj_eff = abs(close - close.shift(1)) / (np.maximum(high, close.shift(1)) - np.minimum(low, close.shift(1)))
    
    # Efficiency persistence
    eff_trend = daily_eff / daily_eff.shift(3)
    
    # High efficiency days count
    high_eff_days = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        high_eff_days.iloc[i] = (daily_eff.iloc[i-4:i+1] > 0.8).sum()
    
    # Efficiency volatility
    eff_vol = daily_eff.rolling(5).std()
    
    # Efficiency-Momentum Blend
    scaled_mom = mom_5d * eff_3d
    
    # Persistent efficiency momentum
    pers_eff_mom = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        window_mom = [mom_5d.iloc[j] * daily_eff.iloc[j] for j in range(i-4, i+1)]
        pers_eff_mom.iloc[i] = sum(window_mom)
    
    # Volume-Confirmed Extreme Reversal
    # 3-day price deviation
    price_dev_3d = pd.Series(index=df.index, dtype=float)
    for i in range(3, len(df)):
        price_change = close.iloc[i] - close.iloc[i-3]
        price_range = max(high.iloc[i-3:i+1].max() - low.iloc[i-3:i+1].min(), 1e-6)
        price_dev_3d.iloc[i] = price_change / price_range
    
    # Volume spike detection
    vol_spike = volume / volume.rolling(5).median()
    
    # Abnormal move
    daily_returns = abs(close - close.shift(1))
    abnormal_move = daily_returns > 2 * daily_returns.rolling(5).std()
    
    # Reversal confirmation
    vol_priced_rev = np.sign(close - close.shift(1)) * volume
    multi_day_rev = np.sign(close.pct_change()) != np.sign(close.pct_change(3))
    extreme_vol_rev = abnormal_move.astype(float) * vol_spike
    
    # Amount Flow Regime Detection
    net_flow = np.sign(close - close.shift(1)) * amount
    flow_mom = net_flow.rolling(3).sum() / amount.rolling(3).sum()
    
    # Flow consistency
    flow_consistency = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        window_flow = net_flow.iloc[i-4:i+1]
        flow_consistency.iloc[i] = (np.sign(window_flow) == np.sign(window_flow.shift(1))).iloc[1:].sum()
    
    # Volatility-Scaled Momentum Convergence
    short_term_vol = close.rolling(5).std()
    medium_term_vol = close.rolling(10).std()
    vol_ratio = short_term_vol / medium_term_vol
    
    # Volatility-adjusted momentum
    daily_ret_vol = close.pct_change().rolling(5).std()
    mom_5d_vol_scaled = mom_5d / daily_ret_vol
    mom_10d_vol_scaled = mom_10d / daily_ret_vol.rolling(10).mean()
    mom_convergence = mom_5d_vol_scaled - mom_10d_vol_scaled
    
    # Combine factors with weights
    factor = (
        0.15 * mom_5d +
        0.12 * mom_ratio +
        0.10 * div_strength +
        0.08 * scaled_mom +
        0.10 * pers_eff_mom +
        0.08 * vol_priced_rev / volume.rolling(10).mean() +
        0.12 * flow_mom +
        0.10 * mom_convergence +
        0.08 * vol_spike * np.sign(mom_5d) +
        0.07 * eff_3d * mom_5d
    )
    
    return factor
