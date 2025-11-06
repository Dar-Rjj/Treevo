import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate basic price components
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr_5'] = df['true_range'].rolling(5).mean()
    
    # Multi-Scale Fractal Efficiency Analysis
    df['intraday_efficiency'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['closing_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    df['efficiency_5ma'] = df['intraday_efficiency'].rolling(5).mean()
    df['efficiency_momentum'] = df['intraday_efficiency'] / (df['efficiency_5ma'] + 1e-8) - 1
    
    # Volume Efficiency Components
    df['volume_5sum'] = df['volume'].rolling(5).sum()
    df['volume_concentration'] = df['volume'] / (df['volume_5sum'] + 1e-8)
    
    # Volume-Price Correlation (5-day)
    df['volume_change'] = df['volume'].pct_change()
    df['price_change'] = df['close'].pct_change()
    df['volume_price_corr'] = df['volume_change'].rolling(5).corr(df['price_change'])
    
    # Volatility Regime & Fractal Classification
    df['vol_5'] = df['returns'].rolling(5).std()
    df['vol_20'] = df['returns'].rolling(20).std()
    df['vol_ratio'] = df['vol_5'] / (df['vol_20'] + 1e-8)
    
    # Market Regime Classification
    df['vol_60_median'] = df['vol_20'].rolling(60).median()
    df['high_vol_regime'] = (df['vol_20'] > df['vol_60_median']).astype(int)
    
    # Hurst-like calculation for low volatility regime
    def hurst_like(series):
        if len(series) < 20:
            return 0.5
        lags = range(2, 10)
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    df['hurst_20'] = df['close'].rolling(20).apply(hurst_like, raw=False)
    df['low_vol_regime'] = ((df['vol_20'] < df['vol_60_median']) & (df['hurst_20'] > 0.55)).astype(int)
    df['transition_regime'] = ((df['high_vol_regime'] == 0) & (df['low_vol_regime'] == 0)).astype(int)
    
    # Breakout Momentum Core
    df['high_5'] = df['high'].rolling(5).max()
    df['low_5'] = df['low'].rolling(5).min()
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    
    df['breakout_5'] = (df['close'] - df['high_5'].shift(1)) / (df['high_5'].shift(1) - df['low_5'].shift(1) + 1e-8)
    df['breakout_20'] = (df['close'] - df['high_20'].shift(1)) / (df['high_20'].shift(1) - df['low_20'].shift(1) + 1e-8)
    
    # Volume Surge Validation
    df['volume_20ma'] = df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > 2 * df['volume_20ma']).astype(int)
    
    # Fractal Breakout Confirmation
    df['gap_size'] = abs(df['open'] - df['close'].shift(1))
    df['gap_absorption'] = df['gap_size'] / (df['atr_5'] + 1e-8)
    
    # Intraday Efficiency & Reversal Patterns
    df['morning_pressure'] = (df['high'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['afternoon_reversal'] = (df['close'] - df['high']) / (df['high'] - df['low'] + 1e-8)
    
    # Price Range Utilization
    df['range_utilization'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
    
    # Efficiency Asymmetry
    df['positive_efficiency'] = df['intraday_efficiency'].where(df['returns'] > 0).rolling(5).mean()
    df['negative_efficiency'] = df['intraday_efficiency'].where(df['returns'] < 0).rolling(5).mean()
    df['efficiency_asymmetry'] = df['positive_efficiency'] - df['negative_efficiency']
    
    # Fractal Synchronization Dynamics
    df['fractal_sync_score'] = (df['volume_price_corr'].abs() * df['efficiency_momentum'].abs()).fillna(0)
    
    # Efficiency Acceleration
    df['efficiency_accel'] = df['intraday_efficiency'].diff(3)
    
    # Adaptive Signal Integration
    # High Volatility Component
    high_vol_component = (df['breakout_20'] * df['volume_surge'] * df['fractal_sync_score'])
    
    # Low Volatility Component  
    low_vol_component = (df['efficiency_momentum'] * df['fractal_sync_score'] * df['efficiency_asymmetry'])
    
    # Transition Component
    transition_component = (df['gap_absorption'] * df['efficiency_accel'] * df['volume_concentration'])
    
    # Regime-specific weighting
    df['regime_multiplier'] = (
        df['high_vol_regime'] * high_vol_component +
        df['low_vol_regime'] * low_vol_component +
        df['transition_regime'] * transition_component
    )
    
    # Core Momentum
    df['core_momentum'] = (
        df['breakout_5'] * 0.3 + 
        df['breakout_20'] * 0.7 + 
        df['efficiency_momentum'] * 0.5
    )
    
    # Final Factor Construction
    df['fractal_factor'] = (
        df['core_momentum'] * 
        df['regime_multiplier'] * 
        df['fractal_sync_score'] * 
        df['volume_concentration'] * 
        (1 - abs(df['afternoon_reversal'])) * 
        np.sign(df['efficiency_accel'])
    )
    
    return df['fractal_factor']
