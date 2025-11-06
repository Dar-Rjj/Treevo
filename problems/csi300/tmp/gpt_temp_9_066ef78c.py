import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum Analysis
    df = df.copy()
    
    # Short-Term Momentum
    df['mom_5d'] = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    df['mom_10d'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10)
    
    # Long-Term Momentum
    df['mom_20d'] = (df['close'] - df['close'].shift(20)) / df['close'].shift(20)
    df['mom_50d'] = (df['close'] - df['close'].shift(50)) / df['close'].shift(50)
    
    # Momentum Divergence Efficiency
    # Calculate efficiency ratios with momentum direction preservation
    df['eff_5_20'] = np.where(
        df['mom_20d'] != 0,
        df['mom_5d'] / df['mom_20d'],
        np.sign(df['mom_5d']) * 100  # Large value when denominator is zero
    )
    
    df['eff_10_50'] = np.where(
        df['mom_50d'] != 0,
        df['mom_10d'] / df['mom_50d'],
        np.sign(df['mom_10d']) * 100
    )
    
    # Combined momentum divergence signal
    momentum_divergence = 0.6 * df['eff_5_20'] + 0.4 * df['eff_10_50']
    
    # Volume Acceleration Dynamics
    df['volume_accel'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Volume trend using linear regression slope over 5 days
    def volume_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    df['volume_trend'] = df['volume'].rolling(window=5, min_periods=5).apply(volume_slope, raw=True)
    
    # Volume breakout efficiency
    df['volume_ratio_20d'] = df['volume'] / df['volume'].rolling(window=20, min_periods=20).mean()
    df['volume_efficiency'] = df['volume_ratio_20d'] * df['volume_accel']
    
    # Intraday Pressure Indicators
    df['intraday_range_eff'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['order_flow_imb'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Volume concentration (simplified - using first available data as proxy)
    df['volume_concentration'] = df['volume'] / df['volume'].rolling(window=5, min_periods=5).sum()
    
    # Volume persistence (consecutive up-volume days)
    volume_up = df['volume'] > df['volume'].shift(1)
    df['volume_persistence'] = volume_up.rolling(window=5, min_periods=1).apply(
        lambda x: x[::-1].cumprod().sum(), raw=False
    )
    
    # Combined Efficiency Scoring
    # Momentum-Volume Efficiency Product
    momentum_volume_product = momentum_divergence * df['volume_efficiency']
    
    # Scale by intraday pressure indicators
    intraday_pressure = (df['intraday_range_eff'] * df['order_flow_imb'] * 
                        df['volume_concentration'] * df['volume_persistence'])
    
    # Conditional signal enhancement
    momentum_volume_efficiency = momentum_volume_product * intraday_pressure
    
    # Apply momentum direction filters
    positive_momentum_mask = (df['mom_5d'] > 0) & (df['mom_10d'] > 0)
    negative_momentum_mask = (df['mom_5d'] < 0) & (df['mom_10d'] < 0)
    
    # Volume confirmation filter
    volume_confirmation = df['volume_efficiency'] > df['volume_efficiency'].rolling(window=20, min_periods=20).mean()
    
    # Final alpha factor with conditional enhancement
    alpha_factor = np.where(
        positive_momentum_mask & volume_confirmation,
        momentum_volume_efficiency * 1.2,  # Enhanced for long signals
        np.where(
            negative_momentum_mask & volume_confirmation,
            momentum_volume_efficiency * 0.8,  # Reduced for short signals
            momentum_volume_efficiency
        )
    )
    
    return pd.Series(alpha_factor, index=df.index)
