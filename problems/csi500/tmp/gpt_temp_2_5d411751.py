import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Pre-calculate rolling statistics for efficiency
    df['volume_ma_5'] = df['volume'].rolling(window=5, min_periods=3).mean()
    df['close_ma_3'] = df['close'].rolling(window=3, min_periods=3).mean()
    df['close_std_3'] = df['close'].rolling(window=3, min_periods=3).std()
    df['high_max_5'] = df['high'].rolling(window=5, min_periods=5).max()
    df['low_min_5'] = df['low'].rolling(window=5, min_periods=5).min()
    
    # Liquidity Dynamics Analysis
    # Volume-Weighted Price Range components
    df['vwap'] = df['amount'] / df['volume']
    df['spread_efficiency'] = (df['high'] - df['low']) / df['vwap']
    
    # Volume-Adjusted Volatility (5-day window)
    vol_adj_volatility = []
    for i in range(len(df)):
        if i >= 4:
            window_high = df['high'].iloc[i-4:i+1]
            window_low = df['low'].iloc[i-4:i+1]
            window_volume = df['volume'].iloc[i-4:i+1]
            numerator = ((window_high - window_low) ** 2 * window_volume).sum()
            denominator = window_volume.sum()
            vol_adj_volatility.append(np.sqrt(numerator / denominator) if denominator > 0 else 0)
        else:
            vol_adj_volatility.append(np.nan)
    df['vol_adj_volatility'] = vol_adj_volatility
    
    # Bid-Ask Spread Proxy components
    df['prev_high_low_range'] = df['high'] - df['low']
    df['prev_high_low_range'] = df['prev_high_low_range'].shift(1)
    df['opening_gap_momentum'] = (df['open'] - df['close'].shift(1)) / df['prev_high_low_range']
    df['intraday_spread_efficiency'] = (df['high'] - df['low']) / (abs(df['close'] - df['open']) + 1e-8)
    
    # Price Reversal Patterns
    # Multi-timeframe Mean Reversion
    df['price_deviation'] = (df['close'] - df['close_ma_3']) / (df['close_std_3'] + 1e-8)
    df['price_oscillation'] = (df['high_max_5'] - df['low_min_5']) / (df['close'] + 1e-8)
    
    # Volume-Confirmed Reversal
    high_volume_mask = df['volume'] > df['volume_ma_5']
    low_volume_mask = df['volume'] < df['volume_ma_5']
    
    df['high_volume_rejection'] = high_volume_mask.astype(float) * (df['high'] - df['close']) / (df['high'] - df['low'] + 1e-8)
    df['low_volume_breakout'] = low_volume_mask.astype(float) * (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['volume_confirmed_reversal'] = df['high_volume_rejection'] - df['low_volume_breakout']
    
    # Liquidity-Momentum Interaction
    # Volume-Weighted Price Momentum (5-day window)
    weighted_return = []
    volume_adjusted_trend = []
    
    for i in range(len(df)):
        if i >= 4:
            window_volume = df['volume'].iloc[i-4:i+1]
            window_close = df['close'].iloc[i-4:i+1]
            window_open = df['open'].iloc[i-4:i+1]
            window_prev_close = df['close'].iloc[i-5:i] if i >= 5 else pd.Series([np.nan] * 5)
            
            # Weighted Return
            returns = (window_close - window_open) / (window_open + 1e-8)
            weighted_ret = (window_volume * returns).sum() / (window_volume.sum() + 1e-8)
            weighted_return.append(weighted_ret)
            
            # Volume-Adjusted Trend
            if i >= 5:
                price_changes = window_close.iloc[1:] - window_prev_close.iloc[-4:]
                vol_trend = (window_volume.iloc[1:] * price_changes).sum() / (window_volume.iloc[1:].sum() + 1e-8)
            else:
                vol_trend = 0
            volume_adjusted_trend.append(vol_trend)
        else:
            weighted_return.append(np.nan)
            volume_adjusted_trend.append(np.nan)
    
    df['weighted_return'] = weighted_return
    df['volume_adjusted_trend'] = volume_adjusted_trend
    
    # Liquidity Regime Detection
    df['high_liquidity_momentum'] = df['weighted_return'] * df['vol_adj_volatility']
    df['low_liquidity_reversal'] = df['price_deviation'] * df['volume_confirmed_reversal']
    
    # Cross-Sectional Liquidity Patterns
    df['intraday_spread_ma_5'] = df['intraday_spread_efficiency'].rolling(window=5, min_periods=3).mean()
    df['relative_spread_efficiency'] = df['intraday_spread_efficiency'] / (df['intraday_spread_ma_5'] + 1e-8)
    df['volume_concentration'] = (df['volume'] / (df['volume_ma_5'] + 1e-8)) * df['vol_adj_volatility']
    
    # Integrated Reversal Momentum
    df['core_reversal_factor'] = df['price_deviation'] * df['volume_confirmed_reversal'] * df['relative_spread_efficiency']
    
    # Regime-Adaptive Enhancement
    # Use volume concentration as regime indicator
    high_liquidity_regime = df['volume_concentration'] > df['volume_concentration'].rolling(window=20, min_periods=10).median()
    low_liquidity_regime = ~high_liquidity_regime
    
    # Final factor calculation
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if pd.notna(df['core_reversal_factor'].iloc[i]):
            if high_liquidity_regime.iloc[i]:
                result.iloc[i] = (df['core_reversal_factor'].iloc[i] * 
                                df['high_liquidity_momentum'].iloc[i] * 
                                df['volume_concentration'].iloc[i])
            else:
                result.iloc[i] = (df['core_reversal_factor'].iloc[i] * 
                                df['low_liquidity_reversal'].iloc[i] / 
                                (abs(df['volume_concentration'].iloc[i]) + 1e-8))
    
    return result
