import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Volatility-Convergence Divergence Factor
    Combines momentum strength, convergence patterns, and volatility regime
    to identify sustainable price movements.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Component
    # Price Momentum
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_divergence'] = data['momentum_3d'] - data['momentum_10d']
    
    # Volume Momentum
    data['volume_change_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_acceleration'] = (data['volume_change_5d'] - 
                                 data['volume_change_5d'].shift(1))
    
    # Volatility Regime Analysis
    data['realized_vol_5d'] = data['close'].pct_change().rolling(5).std()
    data['intraday_vol'] = (data['high'] - data['low']) / data['close']
    
    # 20-day medians for volatility regime classification
    data['realized_vol_median_20d'] = data['realized_vol_5d'].rolling(20).median()
    data['intraday_vol_median_20d'] = data['intraday_vol'].rolling(20).median()
    
    # Volatility Regime
    data['high_vol_regime'] = ((data['realized_vol_5d'] > data['realized_vol_median_20d']) & 
                              (data['intraday_vol'] > data['intraday_vol_median_20d'])).astype(int)
    data['low_vol_regime'] = (~data['high_vol_regime'].astype(bool)).astype(int)
    
    # Convergence Strength
    # Price Convergence
    data['ma_20d'] = data['close'].rolling(20).mean()
    data['price_distance_ma'] = (data['close'] - data['ma_20d']) / data['ma_20d']
    
    # Recent Price Compression
    data['range_3d'] = (data['high'].rolling(3).max() - data['low'].rolling(3).min()) / data['close']
    data['range_10d'] = (data['high'].rolling(10).max() - data['low'].rolling(10).min()) / data['close']
    data['price_compression'] = data['range_3d'] / data['range_10d']
    
    # Volume Convergence
    data['volume_to_range'] = data['volume'] / (data['high'] - data['low'])
    data['volume_stability'] = 1 / (data['volume_to_range'].rolling(5).std() + 1e-8)
    
    # Volume autocorrelation pattern (5-day lag-1 autocorrelation)
    data['volume_autocorr'] = data['volume'].rolling(5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    
    # Composite Scores
    # Momentum Strength (normalized combination)
    data['price_momentum_strength'] = (
        data['momentum_5d'].rolling(10).apply(lambda x: x.rank().iloc[-1] if len(x) == 10 else 0.5, raw=False) +
        (1 - abs(data['momentum_divergence']).rolling(10).apply(lambda x: x.rank().iloc[-1] if len(x) == 10 else 0.5, raw=False))
    ) / 2
    
    data['volume_momentum_strength'] = (
        data['volume_change_5d'].rolling(10).apply(lambda x: x.rank().iloc[-1] if len(x) == 10 else 0.5, raw=False) +
        data['volume_acceleration'].rolling(10).apply(lambda x: x.rank().iloc[-1] if len(x) == 10 else 0.5, raw=False)
    ) / 2
    
    data['momentum_strength'] = (data['price_momentum_strength'] + data['volume_momentum_strength']) / 2
    
    # Convergence Strength
    data['price_convergence'] = (
        (1 - abs(data['price_distance_ma'])).rolling(10).apply(lambda x: x.rank().iloc[-1] if len(x) == 10 else 0.5, raw=False) +
        data['price_compression'].rolling(10).apply(lambda x: x.rank().iloc[-1] if len(x) == 10 else 0.5, raw=False)
    ) / 2
    
    data['volume_convergence'] = (
        data['volume_stability'].rolling(10).apply(lambda x: x.rank().iloc[-1] if len(x) == 10 else 0.5, raw=False) +
        (data['volume_autocorr'] + 1).rolling(10).apply(lambda x: x.rank().iloc[-1] if len(x) == 10 else 0.5, raw=False) / 2
    ) / 2
    
    data['convergence_strength'] = (data['price_convergence'] + data['volume_convergence']) / 2
    
    # Divergence Detection
    data['positive_signal'] = (
        (data['momentum_strength'] > 0.6) & 
        (data['convergence_strength'] > 0.6) & 
        (data['low_vol_regime'] == 1)
    ).astype(int)
    
    data['negative_signal'] = (
        (data['momentum_strength'] < 0.4) & 
        (data['convergence_strength'] < 0.4) & 
        (data['high_vol_regime'] == 1)
    ).astype(int)
    
    # Factor Construction
    # Base Component
    data['base_component'] = data['momentum_strength'] * data['convergence_strength']
    
    # Volatility Adjustment
    data['volatility_adjustment'] = np.where(
        data['low_vol_regime'] == 1, 0.9,
        np.where(data['high_vol_regime'] == 1, 1.1, 1.0)
    )
    
    # Final Factor with Sign
    data['factor'] = data['base_component'] * data['volatility_adjustment']
    
    # Apply divergence detection sign
    data['final_factor'] = np.where(
        data['positive_signal'] == 1, data['factor'],
        np.where(data['negative_signal'] == 1, -data['factor'], 0)
    )
    
    # Normalize final factor
    data['final_factor_normalized'] = (data['final_factor'] - data['final_factor'].rolling(20).mean()) / data['final_factor'].rolling(20).std()
    
    return data['final_factor_normalized']
