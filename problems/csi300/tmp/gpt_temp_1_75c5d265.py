import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate Mid-Price
    data['mid_price'] = (data['high'] + data['low']) / 2
    
    # Multi-Period Momentum Signals
    data['momentum_5d'] = (data['mid_price'] - data['mid_price'].shift(5)) / data['mid_price'].shift(5)
    data['momentum_10d'] = (data['mid_price'] - data['mid_price'].shift(10)) / data['mid_price'].shift(10)
    data['momentum_20d'] = (data['mid_price'] - data['mid_price'].shift(20)) / data['mid_price'].shift(20)
    
    # Directional Volume Flow
    data['price_range'] = data['high'] - data['low']
    data['directional_movement'] = (data['close'] - data['open']) / np.where(data['price_range'] == 0, 1e-10, data['price_range'])
    data['directional_volume_flow'] = data['directional_movement'] * data['volume']
    
    # Multi-Timeframe Volume Momentum
    data['volume_momentum_5d'] = (data['volume'] - data['volume'].shift(5)) / np.where(data['volume'].shift(5) == 0, 1e-10, data['volume'].shift(5))
    data['volume_momentum_10d'] = (data['volume'] - data['volume'].shift(10)) / np.where(data['volume'].shift(10) == 0, 1e-10, data['volume'].shift(10))
    data['volume_momentum_20d'] = (data['volume'] - data['volume'].shift(20)) / np.where(data['volume'].shift(20) == 0, 1e-10, data['volume'].shift(20))
    
    # Multi-Timeframe Synchronization
    data['sync_5d'] = np.sign(data['momentum_5d']) * np.sign(data['volume_momentum_5d'])
    data['sync_10d'] = np.sign(data['momentum_10d']) * np.sign(data['volume_momentum_10d'])
    data['sync_20d'] = np.sign(data['momentum_20d']) * np.sign(data['volume_momentum_20d'])
    
    # Synchronization Patterns
    sync_columns = ['sync_5d', 'sync_10d', 'sync_20d']
    data['positive_sync_count'] = (data[sync_columns] > 0).sum(axis=1)
    data['negative_sync_count'] = (data[sync_columns] < 0).sum(axis=1)
    data['net_sync'] = data['positive_sync_count'] - data['negative_sync_count']
    
    # Calculate Returns for Volatility
    data['returns'] = data['close'].pct_change()
    
    # Bidirectional Volatility
    returns_30d = data['returns'].rolling(window=30, min_periods=15)
    data['upside_vol'] = returns_30d.apply(lambda x: x[x > 0].std() if len(x[x > 0]) > 5 else np.nan)
    data['downside_vol'] = returns_30d.apply(lambda x: x[x < 0].std() if len(x[x < 0]) > 5 else np.nan)
    
    # Volatility Asymmetry
    data['vol_asymmetry'] = (data['upside_vol'] / np.where(data['downside_vol'] == 0, 1e-10, data['downside_vol'])) - 1
    
    # Volatility-Adjusted Momentum
    data['vol_5d'] = data['returns'].rolling(window=5, min_periods=3).std()
    data['vol_10d'] = data['returns'].rolling(window=10, min_periods=5).std()
    data['vol_20d'] = data['returns'].rolling(window=20, min_periods=10).std()
    
    data['momentum_5d_vol_adj'] = data['momentum_5d'] / np.where(data['vol_5d'] == 0, 1e-10, data['vol_5d'])
    data['momentum_10d_vol_adj'] = data['momentum_10d'] / np.where(data['vol_10d'] == 0, 1e-10, data['vol_10d'])
    data['momentum_20d_vol_adj'] = data['momentum_20d'] / np.where(data['vol_20d'] == 0, 1e-10, data['vol_20d'])
    
    # Regime Classification
    data['regime'] = 'neutral'
    data.loc[data['vol_asymmetry'] > 0.2, 'regime'] = 'bull'
    data.loc[data['vol_asymmetry'] < -0.2, 'regime'] = 'bear'
    
    # Price-Volume Efficiency
    data['price_efficiency'] = (data['close'] - data['open']) / np.where(data['price_range'] == 0, 1e-10, data['price_range'])
    data['volume_weighted_efficiency'] = data['price_efficiency'] * data['volume']
    data['efficiency_5d_ma'] = data['volume_weighted_efficiency'].rolling(window=5, min_periods=3).mean()
    
    # Breakout Efficiency
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    data['high_5d_rolling'] = data['high'].rolling(window=5, min_periods=3).max()
    data['low_5d_rolling'] = data['low'].rolling(window=5, min_periods=3).min()
    
    data['breakout_magnitude'] = (
        np.maximum(0, data['high'] - data['high_5d_rolling'].shift(1)) +
        np.maximum(0, data['low_5d_rolling'].shift(1) - data['low'])
    )
    data['efficiency_ratio'] = data['breakout_magnitude'] / np.where(data['true_range'] == 0, 1e-10, data['true_range'])
    
    # Efficiency-Momentum Divergence
    data['efficiency_momentum_divergence'] = (
        data['efficiency_ratio'] * data['momentum_5d'] * 
        (1 - abs(data['efficiency_ratio'].rolling(window=10, min_periods=5).corr(data['momentum_5d'])))
    )
    
    # Intraday Behavior
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_persistence'] = np.sign(data['opening_gap']) * np.sign(data['close'] - data['open'])
    
    # Liquidity Pressure
    data['vw_price_impact'] = data['price_range'] / np.where(data['volume'] == 0, 1e-10, data['volume'])
    data['vw_price_impact_10d_ma'] = data['vw_price_impact'].rolling(window=10, min_periods=5).mean()
    data['vw_price_impact_60d_median'] = data['vw_price_impact'].rolling(window=60, min_periods=30).median()
    data['liquidity_pressure'] = (data['vw_price_impact_10d_ma'] / np.where(data['vw_price_impact_60d_median'] == 0, 1e-10, data['vw_price_impact_60d_median'])) - 1
    
    # Momentum Strength
    data['momentum_strength'] = (
        abs(data['momentum_5d']) + abs(data['momentum_10d']) + abs(data['momentum_20d'])
    ) / 3
    data['momentum_strength_scaled'] = np.cbrt(data['momentum_strength'])
    
    # Volatility Momentum
    data['atr_10d'] = data['true_range'].rolling(window=10, min_periods=5).mean()
    data['volatility_momentum'] = (data['atr_10d'] - data['atr_10d'].shift(5)) / np.where(data['atr_10d'].shift(5) == 0, 1e-10, data['atr_10d'].shift(5))
    
    # Regime-Adaptive Synchronization Score
    def regime_sync_score(row):
        if row['regime'] == 'bull':
            return 0.1 * row['sync_5d'] + 0.3 * row['sync_10d'] + 0.6 * row['sync_20d']
        elif row['regime'] == 'bear':
            return 0.6 * row['sync_5d'] + 0.3 * row['sync_10d'] + 0.1 * row['sync_20d']
        else:
            return 0.33 * row['sync_5d'] + 0.33 * row['sync_10d'] + 0.34 * row['sync_20d']
    
    data['regime_sync_score'] = data.apply(regime_sync_score, axis=1)
    
    # Efficiency Divergence Multiplier
    data['efficiency_multiplier'] = np.cbrt(
        data['efficiency_momentum_divergence'] * 
        (1 + data['gap_persistence'] * data['price_efficiency'])
    )
    
    # Momentum Intensity Adjustment
    data['momentum_intensity'] = (
        data['momentum_strength_scaled'] * 
        (1 - 0.3 * np.tanh(data['liquidity_pressure'])) * 
        (1 + 0.2 * data['volatility_momentum'])
    )
    
    # Final Factor Integration
    data['composite_factor'] = (
        data['regime_sync_score'] * 
        data['efficiency_multiplier'] * 
        data['momentum_intensity']
    )
    
    # Apply bounded output and scaling
    data['final_factor'] = np.tanh(data['composite_factor']) * (1 + 0.5 * abs(data['vol_asymmetry']))
    
    return data['final_factor']
