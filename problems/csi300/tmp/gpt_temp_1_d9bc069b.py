import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    df = data.copy()
    
    # Fractal Momentum Construction
    df['fractal_bull_bear'] = (df['close'] - df['low']) / (df['high'] - df['close']).replace(0, np.nan)
    df['fractal_intraday_eff'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['fractal_gap_momentum'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Multi-Timeframe Efficiency Assessment
    # Short-Term (5-day)
    df['price_movement_5d'] = df['close'] / df['close'].shift(5) - 1
    df['max_high_5d'] = df['high'].rolling(window=5).max()
    df['min_low_5d'] = df['low'].rolling(window=5).min()
    df['volatility_range_5d'] = (df['max_high_5d'] - df['min_low_5d']) / df['close']
    df['efficiency_ratio_5d'] = df['price_movement_5d'] / df['volatility_range_5d'].replace(0, np.nan)
    
    # Medium-Term (20-day)
    df['price_movement_20d'] = df['close'] / df['close'].shift(20) - 1
    df['volatility_range_20d'] = df['close'].diff().abs().rolling(window=20).sum()
    df['efficiency_ratio_20d'] = df['price_movement_20d'] / df['volatility_range_20d'].replace(0, np.nan)
    
    df['fractal_efficiency_regime'] = df['efficiency_ratio_5d'] / df['efficiency_ratio_20d'].replace(0, np.nan)
    
    # Volatility Regime Detection
    df['returns'] = df['close'].pct_change()
    df['vol_std_5d'] = df['returns'].rolling(window=5).std()
    df['vol_std_20d'] = df['returns'].rolling(window=20).std()
    df['volatility_ratio'] = df['vol_std_5d'] / df['vol_std_20d'].replace(0, np.nan)
    
    # Regime Persistence
    df['vol_ratio_gt_prev'] = (df['volatility_ratio'] > df['volatility_ratio'].shift(1)).astype(int)
    df['regime_persistence'] = df['vol_ratio_gt_prev'].rolling(window=5).sum()
    
    # Microstructure Confirmation
    # Volume Fractal Persistence
    df['volume_gt_prev'] = (df['volume'] > df['volume'].shift(1)).astype(int)
    df['volume_fractal_persistence'] = 0
    for i in range(1, len(df)):
        if df['volume_gt_prev'].iloc[i] == 1:
            df['volume_fractal_persistence'].iloc[i] = df['volume_fractal_persistence'].iloc[i-1] + 1
    
    # Amount Efficiency
    df['vwap'] = df['amount'] / df['volume'].replace(0, np.nan)
    df['amount_efficiency'] = (df['high'] - df['low']) / df['vwap'].replace(0, np.nan)
    
    # Microstructure Noise
    df['microstructure_noise'] = abs(df['close'] - df['open']) / df['vwap'].replace(0, np.nan)
    
    # Price-Volume Divergence
    df['short_momentum'] = df['close'] / df['close'].shift(3) - 1
    df['volume_change'] = df['volume'] / df['volume'].shift(1) - 1
    df['price_volume_divergence'] = (np.sign(df['short_momentum']) != np.sign(df['volume_change'])).astype(float)
    
    # Adaptive Momentum Integration
    df['core_fractal_momentum'] = df['fractal_bull_bear'] * df['fractal_intraday_eff']
    
    # Regime-Adaptive Weighting
    df['regime_adaptive_momentum'] = df['core_fractal_momentum']
    high_vol_mask = df['volatility_ratio'] > 1
    df.loc[high_vol_mask, 'regime_adaptive_momentum'] = (
        df['core_fractal_momentum'] * 0.7 + df['fractal_gap_momentum'] * 0.3
    )
    low_vol_mask = df['volatility_ratio'] <= 1
    df.loc[low_vol_mask, 'regime_adaptive_momentum'] = (
        df['core_fractal_momentum'] * (1 + df['regime_persistence'] / 5)
    )
    
    # Efficiency-Enhanced Momentum
    df['efficiency_enhanced_momentum'] = df['regime_adaptive_momentum'] * df['efficiency_ratio_5d']
    
    # Microstructure Filtering
    df['volume_confirmed_factor'] = df['efficiency_enhanced_momentum'] * df['volume_fractal_persistence']
    df['noise_reduced_factor'] = df['volume_confirmed_factor'] * (1 - df['microstructure_noise'])
    df['divergence_filtered_factor'] = df['noise_reduced_factor'] * (1 - df['price_volume_divergence'])
    
    # Hierarchical Alpha Synthesis
    df['primary_adaptive_alpha'] = df['divergence_filtered_factor']
    df['regime_adjusted_alpha'] = df['primary_adaptive_alpha'] * np.sign(df['fractal_efficiency_regime'])
    df['final_alpha'] = df['regime_adjusted_alpha'] * df['amount_efficiency']
    
    return df['final_alpha']
