import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining multiple market microstructure insights
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Price-Range Efficiency Momentum
    # Intraday Range Efficiency
    data['range_efficiency'] = (data['high'] - data['low']) / data['amount']
    data['price_momentum_5d'] = data['close'].pct_change(5)
    data['range_5d_avg'] = (data['high'] - data['low']).rolling(5).mean()
    data['range_expansion'] = (data['high'] - data['low']) / data['range_5d_avg']
    
    # Weight momentum by efficiency and expansion
    data['efficiency_momentum'] = (data['price_momentum_5d'] * 
                                  (1 + data['range_efficiency']) * 
                                  (1 + data['range_expansion']))
    
    # Volume-Cluster Adaptive Momentum
    data['volume_20d_avg'] = data['volume'].rolling(20).mean()
    data['volume_regime'] = (data['volume'] > data['volume_20d_avg']).astype(int)
    
    # Calculate consecutive volume cluster strength
    data['volume_cluster_strength'] = 0
    current_streak = 0
    for i in range(len(data)):
        if data['volume_regime'].iloc[i] == 1:
            current_streak += 1
        else:
            current_streak = 0
        data['volume_cluster_strength'].iloc[i] = current_streak
    
    data['price_acceleration'] = data['close'].pct_change(5).pct_change(3)
    data['close_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Weight momentum by cluster strength
    data['cluster_momentum'] = (data['price_acceleration'] * 
                               (1 + data['volume_cluster_strength'] * 0.1) * 
                               (1 + (data['close_position'] - 0.5) * 2))
    
    # Amount-Weighted Range Reversal
    data['trading_efficiency'] = abs(data['close'].pct_change()) / data['amount']
    data['efficiency_10d_avg'] = data['trading_efficiency'].rolling(10).mean()
    data['efficiency_ratio'] = data['trading_efficiency'] / data['efficiency_10d_avg']
    
    # Volume trend using linear regression slope
    def volume_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    data['volume_trend'] = data['volume'].rolling(5).apply(volume_slope, raw=False)
    
    # Extreme close + low efficiency reversal signal
    extreme_close = ((data['close_position'] > 0.8) | (data['close_position'] < 0.2)).astype(int)
    low_efficiency = (data['efficiency_ratio'] > 1.2).astype(int)
    rising_volume = (data['volume_trend'] > 0).astype(int)
    
    data['reversal_signal'] = (extreme_close * low_efficiency * 
                              (1 + rising_volume * 0.5) * 
                              -np.sign(data['close'].pct_change()))
    
    # Gap-Range Momentum Divergence
    data['opening_gap'] = (data['open'] / data['close'].shift(1) - 1)
    data['intraday_strength'] = data['close_position'] - 0.5
    
    # Gap-Range alignment
    gap_range_alignment = np.sign(data['opening_gap']) * np.sign(data['intraday_strength'])
    data['gap_momentum'] = (data['opening_gap'] * 
                           (1 + gap_range_alignment * 0.5) * 
                           data['volume'] / data['volume_20d_avg'])
    
    # Volatility-Adjusted Amount Efficiency
    data['price_impact_efficiency'] = abs(data['close'].pct_change()) / data['amount']
    data['efficiency_10d_avg_impact'] = data['price_impact_efficiency'].rolling(10).mean()
    data['efficiency_deviation'] = (data['price_impact_efficiency'] / 
                                   data['efficiency_10d_avg_impact'] - 1)
    
    data['volatility_10d'] = (data['high'] - data['low']).rolling(10).mean()
    data['volatility_regime'] = data['volatility_10d'] / data['volatility_10d'].rolling(20).mean()
    
    # Volume concentration (day-over-day change in volume)
    data['volume_concentration'] = data['volume'].pct_change()
    
    # Adjust efficiency signals by volatility
    volatility_adjustment = np.where(data['volatility_regime'] > 1.1, 1.5, 1.0)
    data['volatility_efficiency'] = (data['efficiency_deviation'] * 
                                    volatility_adjustment * 
                                    (1 + data['volume_concentration']))
    
    # Combine all components with weights
    weights = {
        'efficiency_momentum': 0.25,
        'cluster_momentum': 0.25,
        'reversal_signal': 0.20,
        'gap_momentum': 0.15,
        'volatility_efficiency': 0.15
    }
    
    # Final alpha factor
    alpha_factor = (
        weights['efficiency_momentum'] * data['efficiency_momentum'] +
        weights['cluster_momentum'] * data['cluster_momentum'] +
        weights['reversal_signal'] * data['reversal_signal'] +
        weights['gap_momentum'] * data['gap_momentum'] +
        weights['volatility_efficiency'] * data['volatility_efficiency']
    )
    
    return alpha_factor
