import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-timeframe Momentum Deceleration
    data['momentum_5d'] = data['close'].pct_change(5)
    data['momentum_10d'] = data['close'].pct_change(10)
    data['momentum_deceleration'] = (data['momentum_5d'] - data['momentum_10d']) / (np.abs(data['momentum_10d']) + 1e-8)
    
    # Volatility Asymmetry Regime Detection
    returns = data['close'].pct_change()
    positive_returns = returns.where(returns > 0)
    negative_returns = returns.where(returns < 0)
    
    data['upside_vol'] = positive_returns.rolling(window=5, min_periods=3).std()
    data['downside_vol'] = negative_returns.rolling(window=5, min_periods=3).std()
    data['vol_asymmetry_ratio'] = data['upside_vol'] / (data['downside_vol'] + 1e-8)
    
    # Volatility asymmetry persistence
    data['vol_asymmetry_persistence'] = data['vol_asymmetry_ratio'].rolling(window=3).apply(
        lambda x: np.mean(np.diff(x) > 0) if len(x) == 3 else np.nan
    )
    
    # Volume-Price Divergence with Microstructure
    data['volume_3d'] = data['volume'].rolling(window=3).mean()
    data['volume_8d'] = data['volume'].rolling(window=8).mean()
    data['volume_acceleration'] = (data['volume_3d'] - data['volume_8d']) / (data['volume_8d'] + 1e-8)
    
    data['volume_efficiency'] = data['volume'] / (data['high'] - data['low'] + 1e-8)
    
    # Price-volume alignment
    data['price_direction_3d'] = data['close'].rolling(window=3).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0
    )
    data['volume_direction_3d'] = data['volume'].rolling(window=3).apply(
        lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1 if x.iloc[-1] < x.iloc[0] else 0
    )
    data['price_volume_alignment'] = data['price_direction_3d'] * data['volume_direction_3d']
    
    # Trade size distribution
    data['trade_size'] = data['amount'] / (data['volume'] + 1e-8)
    data['trade_size_skew'] = data['trade_size'].rolling(window=5).skew()
    
    # Divergence intensity
    data['divergence_intensity'] = data['momentum_deceleration'] * data['vol_asymmetry_ratio'] * data['price_volume_alignment']
    
    # Intraday Efficiency Patterns
    data['gap_pct'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['opening_efficiency'] = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['closing_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Intraday momentum acceleration
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['open'] + 1e-8)
    data['intraday_momentum_accel'] = data['intraday_momentum'].diff()
    
    # Trade Size Dynamics
    data['trade_size_concentration'] = data['amount'] / (data['volume'] + 1e-8)
    data['trade_size_persistence'] = data['trade_size_concentration'].rolling(window=3).apply(
        lambda x: np.mean(np.abs(np.diff(x)) < 0.1) if len(x) == 3 else np.nan
    )
    
    # Composite Transition Factor Generation
    # Momentum-volatility crossover
    data['momentum_vol_crossover'] = data['momentum_deceleration'] * data['vol_asymmetry_ratio']
    
    # Microstructure-validated divergence
    data['microstructure_divergence'] = (
        data['divergence_intensity'] * 
        data['trade_size_concentration'] * 
        data['closing_efficiency']
    )
    
    # Regime-adaptive weighting
    high_vol_asymmetry = data['vol_asymmetry_ratio'] > data['vol_asymmetry_ratio'].rolling(window=20).quantile(0.7)
    low_vol_asymmetry = data['vol_asymmetry_ratio'] < data['vol_asymmetry_ratio'].rolling(window=20).quantile(0.3)
    
    data['regime_weight'] = np.where(
        high_vol_asymmetry, 
        data['momentum_deceleration'],
        np.where(
            low_vol_asymmetry,
            data['microstructure_divergence'],
            (data['momentum_deceleration'] + data['microstructure_divergence']) / 2
        )
    )
    
    # Final composite signal
    data['composite_signal'] = (
        data['momentum_vol_crossover'] * 0.4 +
        data['microstructure_divergence'] * 0.3 +
        data['regime_weight'] * 0.2 +
        data['intraday_momentum_accel'] * 0.1
    )
    
    # Apply rolling normalization for stationarity
    data['final_factor'] = (
        data['composite_signal'] - 
        data['composite_signal'].rolling(window=20).mean()
    ) / (data['composite_signal'].rolling(window=20).std() + 1e-8)
    
    return data['final_factor']
