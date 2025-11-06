import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility Acceleration components
    # Short-term Volatility Change
    short_term_vol = (df['high'] - df['low']) / df['close']
    short_term_vol_change = short_term_vol - short_term_vol.shift(5)
    
    # Medium-term Volatility Change
    medium_term_vol = df['close'].rolling(window=5).std() / df['close']
    medium_term_vol_change = medium_term_vol - medium_term_vol.shift(5)
    
    # Volatility Regime Shift
    volatility_regime_shift = short_term_vol_change / medium_term_vol_change
    
    # Flow Dynamics components
    # Buy-Flow Change
    buy_flow_current = ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)) * df['volume']
    buy_flow_prev = ((df['close'].shift(5) - df['low'].shift(5)) / 
                     (df['high'].shift(5) - df['low'].shift(5) + 1e-8)) * df['volume'].shift(5)
    buy_flow_change = buy_flow_current - buy_flow_prev
    
    # Sell-Flow Change
    sell_flow_current = ((df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)) * df['volume']
    sell_flow_prev = ((df['high'].shift(5) - df['close'].shift(5)) / 
                      (df['high'].shift(5) - df['low'].shift(5) + 1e-8)) * df['volume'].shift(5)
    sell_flow_change = sell_flow_current - sell_flow_prev
    
    # Net Flow Momentum
    net_flow_momentum = (buy_flow_change - sell_flow_change) / (df['volume'] + 1e-8)
    
    # Range Efficiency components
    # Range Flow Efficiency
    range_flow_efficiency = (abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)) * df['volume']
    
    # Flow Breakout
    rolling_max_high = df['high'].rolling(window=5).max()
    flow_breakout = (df['close'] - rolling_max_high) * ((df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8))
    
    # Factor Integration
    # Volatility-Flow Alignment
    volatility_flow_alignment = volatility_regime_shift * net_flow_momentum
    
    # Enhanced Breakout
    enhanced_breakout = flow_breakout * short_term_vol_change
    
    # Flow Quality
    net_flow_sign = np.sign(net_flow_momentum)
    sign_consistency = net_flow_sign.rolling(window=5).apply(
        lambda x: len(set(x)) == 1 if not x.isnull().any() else np.nan, raw=False
    )
    
    def rolling_corr(x, y, window):
        return pd.Series([x.iloc[i:i+window].corr(y.iloc[i:i+window]) 
                         if i + window <= len(x) and not x.iloc[i:i+window].isnull().any() 
                         and not y.iloc[i:i+window].isnull().any() else np.nan 
                         for i in range(len(x))], index=x.index)
    
    volume_flow_corr = rolling_corr(df['volume'], net_flow_momentum, 5)
    flow_quality = sign_consistency * volume_flow_corr
    
    # Final Composite
    # Core Momentum
    core_momentum = volatility_flow_alignment * enhanced_breakout
    
    # Quality Enhancement
    quality_enhancement = core_momentum * flow_quality
    
    # Adaptive Factor
    adaptive_factor = quality_enhancement * range_flow_efficiency
    
    return adaptive_factor
