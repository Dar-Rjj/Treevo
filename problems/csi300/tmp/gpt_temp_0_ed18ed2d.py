import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Extract OHLCV data
    open_price = df['open']
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Bidirectional Liquidity Components
    # Intraday Liquidity Imbalance
    bid_liquidity_strength = (close - low) / (high - close).replace(0, np.nan)
    ask_liquidity_weakness = (high - close) / (close - low).replace(0, np.nan)
    liquidity_pressure_ratio = ((close - low) * volume) / (high - close).replace(0, np.nan)
    
    # Multi-day Liquidity Accumulation
    bid_momentum_3d = pd.Series(index=df.index, dtype=float)
    ask_resistance_5d = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 2:
            window = slice(i-2, i+1)
            bid_momentum_3d.iloc[i] = ((close.iloc[window] - low.iloc[window]) / 
                                     (high.iloc[window] - close.iloc[window]).replace(0, np.nan)).sum()
        
        if i >= 4:
            window = slice(i-4, i+1)
            ask_resistance_5d.iloc[i] = ((high.iloc[window] - close.iloc[window]) / 
                                       (close.iloc[window] - low.iloc[window]).replace(0, np.nan)).sum()
    
    liquidity_divergence = bid_momentum_3d / ask_resistance_5d.replace(0, np.nan)
    
    # Volume-Liquidity Interaction
    bid_volume_efficiency = ((close - low) * volume) / (high - close).replace(0, np.nan)
    ask_volume_inefficiency = ((high - close) * volume) / (close - low).replace(0, np.nan)
    liquidity_volume_ratio = bid_volume_efficiency / ask_volume_inefficiency.replace(0, np.nan)
    
    # Fractal Liquidity Dynamics
    # Multi-Scale Liquidity Fractals
    short_term_fractal = liquidity_pressure_ratio / liquidity_pressure_ratio.rolling(window=4, min_periods=1).mean().shift(1)
    medium_term_fractal = liquidity_pressure_ratio / liquidity_pressure_ratio.rolling(window=10, min_periods=1).max()
    long_term_stability = liquidity_pressure_ratio / liquidity_pressure_ratio.rolling(window=20, min_periods=1).min()
    
    # Price-Liquidity Fractal Correlation
    price_change_1d = close - close.shift(1)
    liquidity_change_1d = liquidity_pressure_ratio / liquidity_pressure_ratio.shift(1)
    short_term_price_liquidity = price_change_1d / liquidity_change_1d.replace(0, np.nan)
    
    price_change_5d = close - close.shift(5)
    avg_liquidity_4d = liquidity_pressure_ratio.rolling(window=4, min_periods=1).mean().shift(1)
    medium_term_price_liquidity = price_change_5d / (liquidity_pressure_ratio / avg_liquidity_4d).replace(0, np.nan)
    
    # Liquidity Imbalance Momentum
    liquidity_acceleration = (liquidity_pressure_ratio / liquidity_pressure_ratio.shift(1)) / \
                           (liquidity_pressure_ratio.shift(1) / liquidity_pressure_ratio.shift(2)).replace(0, np.nan)
    
    # Bidirectional Liquidity Integration
    # Volatility-Adjusted Liquidity Signals
    bid_liquidity_efficiency = ((close - low) / (high - low).replace(0, np.nan)) * (close > open_price)
    ask_liquidity_efficiency = ((high - close) / (high - low).replace(0, np.nan)) * (close < open_price)
    liquidity_efficiency_ratio = bid_liquidity_efficiency / ask_liquidity_efficiency.replace(0, np.nan)
    
    # Volume-Weighted Liquidity
    volume_ratio = volume / volume.shift(1).replace(0, np.nan)
    volume_confirmed_bid = (close - low) * volume_ratio
    volume_confirmed_ask = (high - close) * volume_ratio
    
    # Multi-Timeframe Liquidity Convergence
    ultra_short_liquidity = (close - low) / (high - close).replace(0, np.nan)
    
    short_term_liquidity = pd.Series(index=df.index, dtype=float)
    medium_term_liquidity = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i >= 3:
            window = slice(i-2, i+1)
            price_change_3d = close.iloc[i] - close.iloc[i-3]
            high_range_3d = high.iloc[window].max() - low.iloc[window].min()
            short_term_liquidity.iloc[i] = price_change_3d / high_range_3d if high_range_3d != 0 else np.nan
        
        if i >= 10:
            window = slice(i-9, i+1)
            price_change_10d = close.iloc[i] - close.iloc[i-10]
            high_range_10d = high.iloc[window].max() - low.iloc[window].min()
            medium_term_liquidity.iloc[i] = price_change_10d / high_range_10d if high_range_10d != 0 else np.nan
    
    # Regime-Adaptive Components
    # Volatility measure
    volatility_5d = (high - low).rolling(window=5, min_periods=1).std()
    avg_volume_5d = volume.rolling(window=5, min_periods=1).mean()
    
    # High Volatility-Low Liquidity Regime
    high_vol_regime = (volatility_5d > volatility_5d.rolling(window=20, min_periods=1).quantile(0.7)) & \
                     (volume < avg_volume_5d)
    
    # Low Volatility-High Liquidity Regime  
    low_vol_regime = (volatility_5d < volatility_5d.rolling(window=20, min_periods=1).quantile(0.3)) & \
                    (volume > avg_volume_5d)
    
    # Transition Regime
    transition_regime = ~high_vol_regime & ~low_vol_regime
    
    # Composite Alpha Factor Construction
    # Volatility-Weighted Liquidity
    volatility_weighted_liquidity = liquidity_efficiency_ratio * liquidity_volume_ratio
    
    # Multi-timeframe Convergence
    timeframe_convergence = (ultra_short_liquidity.fillna(0) + 
                           short_term_liquidity.fillna(0) + 
                           medium_term_liquidity.fillna(0)) / 3
    
    # Regime-classified factors
    high_vol_factor = (volatility_weighted_liquidity * (1 / volume.replace(0, np.nan)) * 
                      liquidity_acceleration).fillna(0)
    
    low_vol_factor = ((volume / volume.rolling(window=5, min_periods=1).min()) * 
                     (1 / (high - low).replace(0, np.nan)) * 
                     ultra_short_liquidity * volume).fillna(0)
    
    transition_factor = ((liquidity_efficiency_ratio / liquidity_efficiency_ratio.shift(3).replace(0, np.nan)) * 
                        (volume / volume.shift(3).replace(0, np.nan)) * 
                        short_term_fractal).fillna(0)
    
    # Final composite factor with regime weighting
    for i in range(len(df)):
        if high_vol_regime.iloc[i]:
            result.iloc[i] = high_vol_factor.iloc[i] * 0.6 + timeframe_convergence.iloc[i] * 0.4
        elif low_vol_regime.iloc[i]:
            result.iloc[i] = low_vol_factor.iloc[i] * 0.7 + timeframe_convergence.iloc[i] * 0.3
        else:
            result.iloc[i] = transition_factor.iloc[i] * 0.5 + timeframe_convergence.iloc[i] * 0.5
    
    # Apply fractal validation - check consistency across timeframes
    fractal_consistency = (short_term_fractal.rolling(window=5, min_periods=1).std() + 
                          medium_term_fractal.rolling(window=5, min_periods=1).std()).fillna(0)
    
    # Final adjustment based on fractal stability
    result = result * (1 / (1 + fractal_consistency))
    
    return result
