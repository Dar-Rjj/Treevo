import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate volatility-adjusted momentum factor with volume confirmation
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # Calculate Price Momentum Component
    # Short-term returns
    data['ret_1d'] = data['close'] / data['close'].shift(1) - 1
    data['ret_2d'] = data['close'] / data['close'].shift(2) - 1
    data['ret_3d'] = data['close'] / data['close'].shift(3) - 1
    
    # Exponentially weighted rolling momentum (5-day window)
    weights = np.exp(np.linspace(-1, 0, 5))
    weights = weights / weights.sum()
    
    def ewma_returns(series):
        return series.rolling(window=5, min_periods=3).apply(
            lambda x: np.sum(x * weights[-len(x):]) if len(x) >= 3 else np.nan
        )
    
    data['momentum_ewma'] = (
        ewma_returns(data['ret_1d']) * 0.5 + 
        ewma_returns(data['ret_2d']) * 0.3 + 
        ewma_returns(data['ret_3d']) * 0.2
    )
    
    # Calculate Volume Confirmation Signal
    # Volume trend analysis
    data['volume_ma_5d'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_accel'] = data['volume'] / data['volume'].shift(1) - 1
    data['volume_accel_ma_3d'] = data['volume_accel'].rolling(window=3, min_periods=2).mean()
    
    # Volume-price relationship
    data['volume_weighted_range'] = data['volume'] * (data['high'] - data['low']) / data['close']
    data['volume_intensity_5d'] = data['volume_weighted_range'].rolling(window=5, min_periods=3).sum()
    data['volume_intensity_ratio'] = data['volume_weighted_range'] / data['volume_intensity_5d']
    
    # Volume confirmation score
    data['volume_confirmation'] = (
        (data['volume'] / data['volume_ma_5d']) * 0.4 +
        (1 + data['volume_accel_ma_3d']) * 0.3 +
        data['volume_intensity_ratio'] * 0.3
    )
    
    # Volatility Adjustment Framework
    # Price volatility measures
    data['volatility_10d'] = data['ret_1d'].rolling(window=10, min_periods=7).std()
    data['daily_range_vol'] = (data['high'] - data['low']) / data['close']
    data['range_vol_ma_5d'] = data['daily_range_vol'].rolling(window=5, min_periods=3).mean()
    
    # Combined volatility measure
    data['combined_volatility'] = (
        data['volatility_10d'].fillna(method='ffill') * 0.6 +
        data['range_vol_ma_5d'] * 0.4
    )
    
    # Risk-adjusted signals
    data['vol_adj_momentum'] = data['momentum_ewma'] / (data['combined_volatility'] + 1e-8)
    data['vol_adj_volume'] = data['volume_confirmation'] / (1 + data['combined_volatility'])
    
    # Feature Interaction and Blending
    # Price gap effects
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_reversal'] = (data['close'] - data['open']) / data['open']
    
    # Multiplicative combination with gap effects
    data['raw_factor'] = (
        data['vol_adj_momentum'] * data['vol_adj_volume'] * 
        (1 + np.sign(data['vol_adj_momentum']) * data['opening_gap']) *
        (1 + data['intraday_reversal'])
    )
    
    # Temporal aggregation with persistence weighting
    recent_perf = data['raw_factor'].rolling(window=3, min_periods=2).mean()
    persistence_weight = 1 + (recent_perf.rolling(window=5, min_periods=3).std() / 
                            (abs(recent_perf.rolling(window=5, min_periods=3).mean()) + 1e-8))
    
    data['final_factor'] = data['raw_factor'].rolling(window=3, min_periods=2).mean() * persistence_weight
    
    # Cross-sectional ranking
    def cross_sectional_rank(series):
        return series.rank(pct=True, method='dense')
    
    # Apply cross-sectional ranking daily
    factor_series = data.groupby(data.index)['final_factor'].transform(cross_sectional_rank)
    
    return factor_series
