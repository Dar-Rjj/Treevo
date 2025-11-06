import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['bid_ask_proxy'] = (data['high'] - data['low']) / data['close']
    
    # Large gap detection and days since last large gap
    large_gap_threshold = 0.02
    data['is_large_gap'] = (data['overnight_gap'].abs() > large_gap_threshold).astype(int)
    
    # Calculate days since last large gap
    data['days_since_large_gap'] = 0
    last_large_gap_idx = -1
    for i in range(len(data)):
        if data['is_large_gap'].iloc[i]:
            last_large_gap_idx = i
        if last_large_gap_idx >= 0:
            data.iloc[i, data.columns.get_loc('days_since_large_gap')] = i - last_large_gap_idx
    
    # Gap momentum decay
    data['gap_momentum_decay'] = data['overnight_gap'] * np.exp(-0.1 * data['days_since_large_gap'])
    
    # Liquidity pressure indicators
    data['volume_median_10d'] = data['volume'].rolling(window=10, min_periods=5).median()
    data['volume_pressure'] = (data['volume'] - data['volume_median_10d']) / data['volume_median_10d']
    data['liquidity_stress'] = data['volume_pressure'] * data['bid_ask_proxy']
    
    # Efficiency persistence metrics
    data['directional_efficiency'] = np.sign(data['close'] - data['close'].shift(1)) * (data['close'] - data['close'].shift(1)) / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency_momentum'] = data['directional_efficiency'].rolling(window=5, min_periods=3).sum()
    
    # Efficiency reversal detection
    data['efficiency_reversal'] = (np.sign(data['directional_efficiency']) != np.sign(data['directional_efficiency'].shift(1))).astype(int)
    
    # Days since efficiency reversal
    data['days_since_efficiency_reversal'] = 0
    last_reversal_idx = -1
    for i in range(len(data)):
        if data['efficiency_reversal'].iloc[i]:
            last_reversal_idx = i
        if last_reversal_idx >= 0:
            data.iloc[i, data.columns.get_loc('days_since_efficiency_reversal')] = i - last_reversal_idx
    
    data['efficiency_persistence'] = data['efficiency_momentum'] * (1 - 0.2 * data['days_since_efficiency_reversal'])
    
    # Integrated factor construction
    data['core_signal'] = data['gap_momentum_decay'] * data['intraday_momentum']
    data['liquidity_adjusted'] = data['core_signal'] * (1 + data['liquidity_stress'])
    data['persistence_weighted'] = data['liquidity_adjusted'] * data['efficiency_persistence']
    
    # Regime detection
    data['volatility_clustering'] = data['bid_ask_proxy'].rolling(window=5, min_periods=3).std()
    data['trend_persistence'] = np.sign(data['close'] - data['close'].shift(1)).rolling(window=10, min_periods=5).sum()
    
    # Regime adjustment factor
    data['regime_adjustment'] = 1.0
    high_vol_threshold = data['volatility_clustering'].quantile(0.7)
    strong_trend_threshold = 6  # 6 out of 10 days same direction
    
    data.loc[(data['volatility_clustering'] > high_vol_threshold) & 
             (data['trend_persistence'].abs() > strong_trend_threshold), 'regime_adjustment'] = 1.3
    
    data.loc[(data['volatility_clustering'] < data['volatility_clustering'].quantile(0.3)) & 
             (data['trend_persistence'].abs() < 3), 'regime_adjustment'] = 0.7
    
    # Final factor with regime adjustment
    data['final_factor'] = data['persistence_weighted'] * data['regime_adjustment']
    
    # Signal classification (for reference, not part of final factor)
    data['high_conviction_long'] = ((data['gap_momentum_decay'] > 0.01) & 
                                   (data['liquidity_stress'] > 0) & 
                                   (data['efficiency_persistence'] > 0.2)).astype(int)
    
    data['high_conviction_short'] = ((data['gap_momentum_decay'] < -0.01) & 
                                    (data['liquidity_stress'] > 0) & 
                                    (data['efficiency_persistence'] < -0.2)).astype(int)
    
    return data['final_factor']
