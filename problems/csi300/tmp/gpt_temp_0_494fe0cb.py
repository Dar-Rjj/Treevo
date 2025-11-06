import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Intraday Volatility-Adjusted Gap Efficiency
    data['overnight_gap'] = data['open'] / data['close'].shift(1) - 1
    data['intraday_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['gap_efficiency'] = data['intraday_efficiency'] / data['overnight_gap'].abs().replace(0, np.nan)
    
    data['intraday_vol'] = (data['high'] - data['low']) / data['open']
    data['vol_percentile'] = data['intraday_vol'].rolling(window=10, min_periods=5).apply(
        lambda x: (x.iloc[-1] > x).mean() if len(x) >= 5 else np.nan
    )
    gap_efficiency_factor = data['gap_efficiency'] * data['vol_percentile']
    
    # Volume-Clustered Momentum Persistence
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5)
    data['volume_regime'] = (data['volume_momentum'] > 1.2).astype(int)
    
    data['price_momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_persistence'] = data['price_momentum_3d'] * data['price_momentum_3d'].shift(1)
    momentum_persistence_factor = data['momentum_persistence'] * data['volume_regime']
    
    # Amount-Weighted Range Breakout Confirmation
    data['breakout_level'] = (data['high'] + data['low']) / 2
    data['breakout_confirmation'] = (data['close'] - data['breakout_level']) / (data['high'] - data['low']).replace(0, np.nan)
    data['breakout_momentum'] = data['breakout_confirmation'].rolling(window=3, min_periods=2).mean()
    
    data['amount_median'] = data['amount'].rolling(window=10, min_periods=5).median()
    data['amount_deviation'] = (data['amount'] - data['amount_median']) / data['amount_median'].replace(0, np.nan)
    data['amount_confidence'] = data['amount_deviation'].abs()
    breakout_factor = data['breakout_momentum'] * data['amount_confidence']
    
    # Volatility-Compressed Trend Acceleration
    data['daily_range'] = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2)
    data['avg_range_5d'] = data['daily_range'].rolling(window=5, min_periods=3).mean()
    data['compression_ratio'] = data['daily_range'] / data['avg_range_5d']
    
    data['price_change_3d'] = data['close'] - data['close'].shift(3)
    data['price_change_6d'] = data['close'] - data['close'].shift(6)
    data['acceleration'] = data['price_change_3d'] - data['price_change_6d']
    compression_factor = data['acceleration'] * (1 - data['compression_ratio'])
    
    # Volume-Profile Adjusted Mean Reversion Efficiency
    data['avg_close_5d'] = data['close'].rolling(window=5, min_periods=3).mean()
    data['price_deviation'] = data['close'] / data['avg_close_5d'] - 1
    data['reversion_strength'] = -data['price_deviation'] * data['price_deviation'].abs()
    
    data['volume_percentile_20d'] = data['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > x).mean() if len(x) >= 10 else np.nan
    )
    data['volume_profile'] = data['volume_percentile_20d'] / 100
    reversion_factor = data['reversion_strength'] * data['volume_profile']
    
    # Intraday Momentum-Volume Convergence
    data['close_position_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['momentum_3d_avg'] = data['close_position_momentum'].rolling(window=3, min_periods=2).mean()
    data['momentum_divergence'] = data['close_position_momentum'] - data['momentum_3d_avg']
    
    data['volume_change'] = data['volume'] / data['volume'].shift(1)
    data['volume_momentum_convergence'] = data['momentum_divergence'] * data['volume_change']
    convergence_factor = data['volume_momentum_convergence'] * data['volume_momentum_convergence'].abs()
    
    # Range-Efficiency Adjusted Momentum Persistence
    data['price_efficiency'] = (data['close'] - data['open']).abs() / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency_avg_5d'] = data['price_efficiency'].rolling(window=5, min_periods=3).mean()
    data['efficiency_ratio'] = data['price_efficiency'] / data['efficiency_avg_5d']
    
    data['return_momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_persistence_adj'] = data['return_momentum_3d'] * data['return_momentum_3d'].shift(1)
    efficiency_factor = data['momentum_persistence_adj'] * data['efficiency_ratio']
    
    # Amount-Volatility Regime Breakout
    data['volatility_5d'] = ((data['high'] - data['low']) / data['close']).rolling(window=5, min_periods=3).mean()
    data['volatility_median_20d'] = data['volatility_5d'].rolling(window=20, min_periods=10).median()
    data['regime_shift'] = data['volatility_5d'] / data['volatility_median_20d']
    
    data['amount_avg_5d'] = data['amount'].rolling(window=5, min_periods=3).mean()
    data['amount_momentum'] = data['amount'] / data['amount_avg_5d']
    data['regime_amount_interaction'] = data['regime_shift'] * data['amount_momentum']
    regime_factor = data['regime_amount_interaction'] * data['regime_amount_interaction'].abs()
    
    # Volume-Weighted Gap Fade Efficiency
    data['gap_fade'] = (data['close'] - data['open']) / data['open']
    data['fade_efficiency'] = data['gap_fade'] / data['overnight_gap'].replace(0, np.nan)
    
    data['volume_avg_10d'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_ratio'] = data['volume'] / data['volume_avg_10d']
    data['volume_weighted_efficiency'] = data['fade_efficiency'] * data['volume_ratio']
    fade_factor = data['volume_weighted_efficiency'] * data['volume_weighted_efficiency'].abs()
    
    # Momentum-Cluster Range Expansion
    data['price_momentum_3d_cluster'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_std_20d'] = data['price_momentum_3d_cluster'].rolling(window=20, min_periods=10).std()
    data['momentum_cluster'] = (data['price_momentum_3d_cluster'].abs() > data['momentum_std_20d']).astype(int)
    
    data['range_avg_5d'] = (data['high'] - data['low']).rolling(window=5, min_periods=3).mean()
    data['range_expansion'] = (data['high'] - data['low']) / data['range_avg_5d']
    data['cluster_expansion'] = data['momentum_cluster'] * data['range_expansion']
    expansion_factor = data['cluster_expansion'] * np.sign(data['price_momentum_3d_cluster'])
    
    # Combine all factors with equal weights
    factors = [
        gap_efficiency_factor,
        momentum_persistence_factor,
        breakout_factor,
        compression_factor,
        reversion_factor,
        convergence_factor,
        efficiency_factor,
        regime_factor,
        fade_factor,
        expansion_factor
    ]
    
    # Normalize each factor by its rolling standard deviation
    combined_factor = pd.Series(0, index=data.index)
    for factor in factors:
        factor_std = factor.rolling(window=20, min_periods=10).std()
        normalized_factor = factor / factor_std.replace(0, np.nan)
        combined_factor = combined_factor + normalized_factor.fillna(0)
    
    return combined_factor
