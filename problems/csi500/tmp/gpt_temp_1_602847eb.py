import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Aware Volume-Weighted Momentum factor
    Combines multi-timeframe momentum with volume confirmation and volatility-adjusted range,
    adapting weights based on market regime detection.
    """
    
    # Multi-Timeframe Momentum
    df['momentum_5d'] = df['close'] / df['close'].shift(5) - 1
    df['momentum_20d'] = df['close'] / df['close'].shift(20) - 1
    df['momentum_60d'] = df['close'] / df['close'].shift(60) - 1
    
    # Volume Confirmation Signals
    df['volume_strength_5d'] = df['volume'] / df['volume'].shift(5)
    df['volume_vs_avg_20d'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Volatility-Adjusted Range
    df['daily_range'] = (df['high'] - df['low']) / df['close'].shift(1)
    
    # Calculate return volatility (20-day)
    returns = df['close'].pct_change()
    df['volatility_20d'] = returns.rolling(window=20).std()
    
    # Market Regime Detection (using close as proxy for market)
    df['market_return_20d'] = df['close'] / df['close'].shift(20) - 1
    market_returns = df['close'].pct_change()
    df['market_volatility_60d'] = market_returns.rolling(window=60).std()
    
    # Define regime thresholds
    high_vol_threshold = df['market_volatility_60d'].quantile(0.7)
    low_vol_threshold = df['market_volatility_60d'].quantile(0.3)
    bull_threshold = 0.02  # 2% return over 20 days
    bear_threshold = -0.02  # -2% return over 20 days
    
    # Initialize regime columns
    df['is_high_vol'] = df['market_volatility_60d'] > high_vol_threshold
    df['is_low_vol'] = df['market_volatility_60d'] < low_vol_threshold
    df['is_bull'] = df['market_return_20d'] > bull_threshold
    df['is_bear'] = df['market_return_20d'] < bear_threshold
    
    # Adaptive Weighting based on regimes
    factor_values = []
    
    for i in range(len(df)):
        if i < 60:  # Ensure enough data for calculations
            factor_values.append(np.nan)
            continue
            
        row = df.iloc[i]
        
        # Base weights
        momentum_weight = 0.4
        volume_weight = 0.3
        range_weight = 0.3
        
        # Regime-based adjustments
        if row['is_high_vol']:
            # High Volatility: Emphasize volume confirmation, reduce momentum
            momentum_weight *= 0.7
            volume_weight *= 1.3
            range_weight *= 1.0
            
        if row['is_low_vol']:
            # Low Volatility: Emphasize momentum, reduce range
            momentum_weight *= 1.3
            volume_weight *= 1.0
            range_weight *= 0.7
            
        if row['is_bull']:
            # Bull Market: Favor long-term momentum, reduce short-term noise
            momentum_components = (
                row['momentum_5d'] * 0.2 + 
                row['momentum_20d'] * 0.3 + 
                row['momentum_60d'] * 0.5
            )
            
        elif row['is_bear']:
            # Bear Market: Favor short-term signals, increase volume confirmation
            momentum_components = (
                row['momentum_5d'] * 0.5 + 
                row['momentum_20d'] * 0.3 + 
                row['momentum_60d'] * 0.2
            )
            volume_weight *= 1.2
        else:
            # Neutral regime
            momentum_components = (
                row['momentum_5d'] * 0.33 + 
                row['momentum_20d'] * 0.33 + 
                row['momentum_60d'] * 0.34
            )
        
        # Volume components
        volume_components = (
            np.log1p(row['volume_strength_5d']) * 0.6 + 
            np.log1p(row['volume_vs_avg_20d']) * 0.4
        )
        
        # Range component (inverse relationship - lower range is better for momentum)
        range_component = -row['daily_range'] / (row['volatility_20d'] + 1e-8)
        
        # Combine components with regime-adjusted weights
        total_factor = (
            momentum_components * momentum_weight +
            volume_components * volume_weight +
            range_component * range_weight
        )
        
        factor_values.append(total_factor)
    
    # Create output series
    factor_series = pd.Series(factor_values, index=df.index, name='regime_aware_volume_momentum')
    
    return factor_series
