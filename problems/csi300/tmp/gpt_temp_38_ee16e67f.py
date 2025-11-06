import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Gap Asymmetry Analysis
    # Overnight gap momentum
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Gap direction persistence (4-day window)
    gap_sign = np.sign(data['overnight_gap'])
    gap_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        window = gap_sign.iloc[i-4:i+1]
        gap_persistence.iloc[i] = (window == window.shift(1)).sum() - 1  # Exclude first comparison
    
    # Gap filling efficiency
    data['gap_fill_eff'] = (data['close'] - np.minimum(data['open'], data['close'].shift(1))) / \
                          (np.maximum(data['open'], data['close'].shift(1)) - np.minimum(data['open'], data['close'].shift(1)))
    data['gap_fill_eff'] = data['gap_fill_eff'].replace([np.inf, -np.inf], np.nan)
    
    # Intraday gap asymmetry
    data['intraday_asym'] = (data['high'] - data['open']) - (data['open'] - data['low'])
    
    # Volatility-Adaptive Signal Processing
    # 10-day return volatility
    data['ret_vol_10d'] = data['close'].pct_change().rolling(window=10, min_periods=5).std()
    
    # Volatility momentum scaling
    data['vol_momentum'] = data['ret_vol_10d'] / data['ret_vol_10d'].shift(5) - 1
    
    # Rolling volatility adjustment using High-Low ranges
    data['hl_range_vol'] = (data['high'] - data['low']).rolling(window=10, min_periods=5).std()
    
    # Volatility-weighted reversion (3-day sum)
    vol_weighted_rev = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])) * \
                      (data['volume'] / data['volume'].shift(1))
    data['vol_weighted_rev_3d'] = vol_weighted_rev.rolling(window=3, min_periods=2).sum()
    
    # Liquidity-Volume Confirmation Framework
    # Gap volume intensity
    data['gap_vol_intensity'] = data['volume'] * abs(data['overnight_gap'])
    
    # Volume trend strength (4-day window)
    vol_trend = pd.Series(index=data.index, dtype=float)
    for i in range(4, len(data)):
        window = data['volume'].iloc[i-4:i+1]
        vol_trend.iloc[i] = (window > window.shift(1)).sum() - 1  # Exclude first comparison
    
    # Intraday volume skew (5-day window)
    data['intraday_vol_skew'] = 0.0
    for i in range(5, len(data)):
        window = data.iloc[i-4:i+1]
        up_vol = window[window['close'] > window['open']]['volume'].sum()
        down_vol = window[window['close'] < window['open']]['volume'].sum()
        data['intraday_vol_skew'].iloc[i] = up_vol - down_vol
    
    # Liquidity momentum
    data['vwap_t'] = data['amount'] / data['volume']
    data['vwap_t_5'] = data['amount'].shift(5) / data['volume'].shift(5)
    data['liquidity_momentum'] = data['vwap_t'] - data['vwap_t_5']
    
    # Cross-Sectional Signal Validation (using rolling quantiles as proxy)
    data['rel_gap_momentum'] = data['overnight_gap'] / data['overnight_gap'].rolling(window=20, min_periods=10).mean()
    data['vol_divergence'] = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    data['gap_recovery_rank'] = data['gap_fill_eff'].rolling(window=20, min_periods=10).rank(pct=True)
    
    # Dynamic Factor Integration
    # Volatility-weighted gap momentum signals
    vol_weighted_gap = data['overnight_gap'] / (data['ret_vol_10d'] + 1e-8)
    
    # Volume-confirmed gap filling patterns
    vol_confirmed_gap = data['gap_fill_eff'] * (data['volume'] / data['volume'].rolling(window=10, min_periods=5).mean())
    
    # Liquidity-adjusted reversion strength
    liquidity_adj_rev = data['vol_weighted_rev_3d'] * data['liquidity_momentum']
    
    # Multi-timeframe signal consistency
    signal_consistency = (gap_persistence.fillna(0) + vol_trend.fillna(0)) / 2
    
    # Final factor combination
    factor = (
        0.3 * vol_weighted_gap +
        0.25 * vol_confirmed_gap +
        0.2 * liquidity_adj_rev +
        0.15 * signal_consistency +
        0.1 * data['intraday_asym']
    )
    
    # Clean and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    return factor
