import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility-Adaptive Price-Volume Efficiency Divergence factor
    """
    data = df.copy()
    
    # True Range Volatility Components
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Multi-timeframe volatility
    data['tr_5d_avg'] = data['true_range'].rolling(window=5, min_periods=3).mean()
    data['tr_20d_avg'] = data['true_range'].rolling(window=20, min_periods=10).mean()
    data['volatility_ratio'] = data['tr_5d_avg'] / data['tr_20d_avg']
    
    # Volatility regime classification
    data['vol_regime'] = 1  # Normal
    data.loc[data['volatility_ratio'] > 1.2, 'vol_regime'] = 2  # High
    data.loc[data['volatility_ratio'] < 0.8, 'vol_regime'] = 0  # Low
    
    # Volatility persistence
    data['regime_persistence'] = 0
    for i in range(1, len(data)):
        if data['vol_regime'].iloc[i] == data['vol_regime'].iloc[i-1]:
            data['regime_persistence'].iloc[i] = data['regime_persistence'].iloc[i-1] + 1
    
    # Price Efficiency Calculations
    data['daily_return'] = data['close'].pct_change()
    data['abs_daily_return'] = abs(data['daily_return'])
    
    # Short-term efficiency (5-day)
    data['price_change_5d'] = abs(data['close'] / data['close'].shift(5) - 1)
    data['cumulative_abs_5d'] = data['abs_daily_return'].rolling(window=5, min_periods=3).sum()
    data['eff_short'] = data['price_change_5d'] / (data['cumulative_abs_5d'] + 1e-8)
    
    # Medium-term efficiency (10-day)
    data['price_change_10d'] = abs(data['close'] / data['close'].shift(10) - 1)
    data['cumulative_abs_10d'] = data['abs_daily_return'].rolling(window=10, min_periods=5).sum()
    data['eff_medium'] = data['price_change_10d'] / (data['cumulative_abs_10d'] + 1e-8)
    
    # Intraday price efficiency
    data['intraday_range'] = data['high'] - data['low']
    data['intraday_move'] = abs(data['close'] - data['open'])
    data['eff_intraday'] = data['intraday_move'] / (data['intraday_range'] + 1e-8)
    
    # Volume Profile Asymmetry Analysis
    data['up_day'] = (data['close'] > data['open']).astype(int)
    data['down_day'] = (data['close'] < data['open']).astype(int)
    
    # Volume concentration ratios
    for window in [3, 8, 13]:
        data[f'up_volume_{window}d'] = (data['volume'] * data['up_day']).rolling(window=window, min_periods=window//2).sum()
        data[f'down_volume_{window}d'] = (data['volume'] * data['down_day']).rolling(window=window, min_periods=window//2).sum()
        data[f'volume_concentration_{window}d'] = (data[f'up_volume_{window}d'] - data[f'down_volume_{window}d']) / (data[f'up_volume_{window}d'] + data[f'down_volume_{window}d'] + 1e-8)
    
    # Liquidity Momentum
    data['amount_3d_avg'] = data['amount'].rolling(window=3, min_periods=2).mean()
    data['amount_8d_avg'] = data['amount'].rolling(window=8, min_periods=4).mean()
    data['liquidity_momentum'] = (data['amount_3d_avg'] / data['amount_8d_avg']) - 1
    
    # Volume-Price Efficiency Mismatch
    data['volume_return_ratio'] = data['volume'] / (abs(data['daily_return']) + 1e-8)
    data['volume_efficiency_5d'] = data['volume_return_ratio'] / data['volume_return_ratio'].rolling(window=5, min_periods=3).mean()
    
    # Multi-Timeframe Momentum
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Directional Consistency Analysis
    data['return_sign'] = np.sign(data['daily_return'])
    data['momentum_5d_sign'] = np.sign(data['momentum_5d'])
    
    directional_consistency = []
    for i in range(len(data)):
        if i < 5:
            directional_consistency.append(0)
        else:
            window_returns = data['return_sign'].iloc[i-4:i+1]
            target_sign = data['momentum_5d_sign'].iloc[i]
            consistency = (window_returns == target_sign).sum() / 5
            directional_consistency.append(consistency)
    data['directional_consistency'] = directional_consistency
    
    # Momentum Stability
    data['return_variance_5d'] = data['daily_return'].rolling(window=5, min_periods=3).var()
    data['momentum_stability'] = abs(data['momentum_5d']) / (data['return_variance_5d'] + 0.0001)
    
    # Multi-scale momentum alignment
    momentum_signs = np.sign(data[['momentum_3d', 'momentum_5d', 'momentum_10d']])
    data['momentum_alignment'] = (momentum_signs.sum(axis=1) / 3).abs()
    
    # Structural Break Detection - Recent High/Low
    data['recent_high'] = data['high'].rolling(window=20, min_periods=10).max()
    data['recent_low'] = data['low'].rolling(window=20, min_periods=10).min()
    data['break_high'] = (data['close'] > data['recent_high'].shift(1)).astype(int)
    data['break_low'] = (data['close'] < data['recent_low'].shift(1)).astype(int)
    
    # Volume confirmation for breaks
    data['volume_break_ratio'] = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    
    # Composite Factor Construction
    
    # Base Efficiency-Momentum Core
    data['base_efficiency'] = (0.4 * data['eff_short'] + 0.4 * data['eff_medium'] + 0.2 * data['eff_intraday'])
    data['base_efficiency'] *= np.sign(data['momentum_5d'])
    data['base_efficiency'] *= (0.5 + 0.5 * data['directional_consistency'])
    data['base_efficiency'] *= (0.3 + 0.7 * data['momentum_stability'].clip(0, 2))
    
    # Volume-Liquidity Enhancement
    volume_component = (0.4 * data['volume_concentration_8d'] + 
                       0.3 * data['volume_efficiency_5d'].clip(-2, 2) + 
                       0.3 * data['liquidity_momentum'])
    
    # Volatility Regime Adaptation
    regime_weights = np.select([
        data['vol_regime'] == 0,  # Low volatility
        data['vol_regime'] == 1,  # Normal volatility
        data['vol_regime'] == 2   # High volatility
    ], [
        # Low volatility: emphasize momentum quality
        0.6 * data['base_efficiency'] + 0.2 * volume_component + 0.2 * data['momentum_alignment'],
        # Normal volatility: balanced approach
        0.5 * data['base_efficiency'] + 0.3 * volume_component + 0.2 * data['momentum_alignment'],
        # High volatility: efficiency focus with volume confirmation
        0.7 * data['base_efficiency'] + 0.3 * volume_component
    ])
    
    # Regime persistence adjustment
    persistence_boost = 1 + 0.1 * np.minimum(data['regime_persistence'] / 10, 1)
    regime_weights *= persistence_boost
    
    # Divergence-Based Signal Enhancement
    price_volume_divergence = (data['momentum_5d'] - data['volume_concentration_8d'] * 2).clip(-0.2, 0.2)
    
    # Breakout confirmation
    breakout_strength = np.select([
        (data['break_high'] | data['break_low']) & (data['volume_break_ratio'] > 1.5),
        (data['break_high'] | data['break_low']) & (data['volume_break_ratio'] > 1.0),
        (data['break_high'] | data['break_low']) & (data['volume_break_ratio'] <= 1.0)
    ], [1.2, 1.0, 0.8], default=1.0)
    
    # Final composite factor
    composite_factor = (regime_weights * 
                       (1 + 0.5 * price_volume_divergence) * 
                       breakout_strength)
    
    # Normalize and clean
    composite_factor = composite_factor.replace([np.inf, -np.inf], np.nan)
    composite_factor = (composite_factor - composite_factor.rolling(window=50, min_periods=20).mean()) / composite_factor.rolling(window=50, min_periods=20).std()
    
    return composite_factor
