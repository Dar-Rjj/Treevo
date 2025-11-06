import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Calculate daily range
    df['daily_range'] = df['high'] - df['low']
    
    # Volatility Regime Classification
    df['range_5d_avg'] = df['daily_range'].rolling(window=5).mean()
    df['range_20d_median'] = df['daily_range'].rolling(window=20).median()
    
    # Create volatility regime indicator
    df['vol_regime'] = 1  # Normal volatility by default
    high_vol_condition = df['range_5d_avg'] > (1.3 * df['range_20d_median'])
    low_vol_condition = df['range_5d_avg'] < (0.8 * df['range_20d_median'])
    df.loc[high_vol_condition, 'vol_regime'] = 2  # High volatility
    df.loc[low_vol_condition, 'vol_regime'] = 0   # Low volatility
    
    # Regime-Adaptive Momentum Acceleration
    def calculate_momentum_acceleration(close_prices, window):
        """Calculate momentum acceleration as second derivative of price"""
        momentum = close_prices.diff(window)
        acceleration = momentum.diff(1)
        return acceleration
    
    # Calculate momentum acceleration based on regime
    df['momentum_acc'] = np.nan
    high_vol_mask = df['vol_regime'] == 2
    low_vol_mask = df['vol_regime'] == 0
    normal_vol_mask = df['vol_regime'] == 1
    
    df.loc[high_vol_mask, 'momentum_acc'] = calculate_momentum_acceleration(df['close'], 3)
    df.loc[low_vol_mask, 'momentum_acc'] = calculate_momentum_acceleration(df['close'], 8)
    df.loc[normal_vol_mask, 'momentum_acc'] = calculate_momentum_acceleration(df['close'], 5)
    
    # Volume Confirmation - Volume-range alignment
    df['volume_5d_avg'] = df['volume'].rolling(window=5).mean()
    df['range_5d_avg'] = df['daily_range'].rolling(window=5).mean()
    
    # Calculate 5-day correlation between volume and range
    volume_range_corr = []
    for i in range(len(df)):
        if i >= 4:
            vol_window = df['volume'].iloc[i-4:i+1]
            range_window = df['daily_range'].iloc[i-4:i+1]
            corr = vol_window.corr(range_window)
            volume_range_corr.append(corr if not np.isnan(corr) else 0)
        else:
            volume_range_corr.append(0)
    df['volume_range_corr'] = volume_range_corr
    
    # Volume Confirmation - Volume-price divergence
    # Calculate VWAP (Volume Weighted Average Price)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (df['typical_price'] * df['volume']).rolling(window=8).sum() / df['volume'].rolling(window=8).sum()
    
    # Calculate 8-day slopes
    df['price_slope'] = df['close'].diff(8) / 8
    df['vwap_slope'] = df['vwap'].diff(8) / 8
    
    # Volume-price divergence
    df['volume_price_divergence'] = df['price_slope'] - df['vwap_slope']
    
    # Signal Integration
    # Combine acceleration with volume confirmation
    df['volume_confirmation'] = (df['volume_range_corr'] > 0.3) & (df['volume_price_divergence'] > 0)
    df['raw_signal'] = df['momentum_acc'] * df['volume_confirmation'].astype(float)
    
    # Apply momentum persistence filter (3-day consistency)
    df['signal_3d_avg'] = df['raw_signal'].rolling(window=3).mean()
    df['signal_3d_std'] = df['raw_signal'].rolling(window=3).std()
    
    # Final factor: raw signal weighted by consistency (inverse of std)
    df['factor'] = df['raw_signal'] * (1 / (df['signal_3d_std'] + 1e-6))
    
    return df['factor']
