import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Momentum-Volume Asymmetry Divergence Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Momentum Asymmetry Analysis
    # Short-term momentum asymmetry
    data['price_momentum_div'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1) - \
                                (data['close'].shift(5) - data['close'].shift(6)) / data['close'].shift(6)
    
    data['range_efficiency_momentum'] = (data['high'] - data['open']) / (data['high'] - data['low'] + 1e-8) - \
                                       (data['open'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    data['opening_gap_momentum'] = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    
    # Medium-term momentum patterns
    data['momentum_div_ratio'] = ((data['close'] - data['close'].shift(5)) / (data['close'].shift(5) + 1e-8)) / \
                                ((data['close'] - data['close'].shift(10)) / (data['close'].shift(10) + 1e-8) + 1e-8)
    
    # Calculate daily range efficiency
    data['daily_range_efficiency'] = (data['high'] - data['low']) / (data['close'] + 1e-8)
    data['efficiency_persistence'] = data['daily_range_efficiency'].rolling(window=5, min_periods=3).corr(
        pd.Series(range(5), index=data.index[-5:] if len(data) >= 5 else data.index))
    
    # Momentum quality assessment
    data['momentum_trend'] = data['close'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else np.nan, raw=True)
    
    # Momentum regime detection
    data['momentum_second_deriv'] = data['close'].diff().diff()
    
    # Volume Asymmetry and Pressure Analysis
    # Volume concentration asymmetry
    data['up_day'] = data['close'] > data['open']
    data['down_day'] = data['close'] < data['open']
    
    data['up_day_volume_conc'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: x[data['up_day'].iloc[-len(x):].values].mean() if sum(data['up_day'].iloc[-len(x):]) > 0 else 0, raw=False)
    
    data['down_day_volume_conc'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: x[data['down_day'].iloc[-len(x):].values].mean() if sum(data['down_day'].iloc[-len(x):]) > 0 else 0, raw=False)
    
    data['volume_asymmetry_ratio'] = data['up_day_volume_conc'] / (data['down_day_volume_conc'] + 1e-8)
    
    # Volume momentum and trends
    data['volume_ratio'] = data['volume'] / (data['volume'].shift(1) + 1e-8)
    data['volume_trend'] = data['volume'] / (data['volume'].shift(5) + 1e-8)
    data['volume_persistence'] = (data['volume'] / (data['volume'].shift(1) + 1e-8)) * \
                                (data['volume'].shift(1) / (data['volume'].shift(2) + 1e-8))
    
    # Volume-pressure integration
    data['volume_concentration'] = data['volume'] * abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    short_term_momentum = (data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    medium_term_momentum = (data['close'] - data['close'].shift(5)) / (data['close'].shift(5) + 1e-8)
    data['volume_weighted_momentum'] = data['volume'] * (short_term_momentum + medium_term_momentum)
    
    # Volatility-Price Structure Integration
    # Multi-period directional volatility
    data['upside_volatility'] = (data['high'] - data['close']).rolling(window=3, min_periods=2).mean()
    data['downside_volatility'] = (data['close'] - data['low']).rolling(window=3, min_periods=2).mean()
    data['volatility_asymmetry'] = data['upside_volatility'] / (data['downside_volatility'] + 1e-8)
    
    # True Range calculations
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['true_range_5d'] = data['true_range'].rolling(window=5, min_periods=3).mean()
    data['true_range_20d'] = data['true_range'].rolling(window=20, min_periods=10).mean()
    
    # Price-pressure components
    data['closing_strength'] = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    
    # Divergence Detection and Synchronization
    # Momentum-volume divergence
    momentum_acceleration = data['close'].pct_change().diff()
    volume_trend_slope = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 3 else np.nan, raw=True)
    
    data['momentum_volume_div'] = momentum_acceleration * volume_trend_slope
    
    # Volatility-volume divergence
    true_range_acceleration = data['true_range'].diff()
    data['volatility_volume_div'] = true_range_acceleration * data['volume_persistence']
    
    # Price-volume synchronization
    data['positive_momentum_high_volume'] = ((data['close'] > data['open']) & 
                                           (data['volume'] > data['volume'].rolling(window=10, min_periods=5).mean())).astype(int)
    
    data['negative_momentum_low_volume'] = ((data['close'] < data['open']) & 
                                          (data['volume'] < data['volume'].rolling(window=10, min_periods=5).mean())).astype(int)
    
    # Persistence of divergence
    data['divergence_persistence'] = data['momentum_volume_div'].rolling(window=5, min_periods=3).apply(
        lambda x: sum(np.diff(np.sign(x)) != 0) if len(x) >= 3 else 0, raw=True)
    
    # Final Alpha Factor Construction
    # Multi-timeframe signal integration
    short_term_div = data['momentum_volume_div'].rolling(window=3, min_periods=2).mean()
    medium_term_div = data['momentum_volume_div'].rolling(window=10, min_periods=5).mean()
    
    volume_confirmation = data['volume_asymmetry_ratio'] * data['volume_persistence']
    
    # Asymmetry strength assessment
    momentum_asymmetry = abs(data['range_efficiency_momentum']) * data['volatility_asymmetry']
    volume_validation = data['volume_concentration'] * data['volume_weighted_momentum']
    
    # Momentum exhaustion as reversal triggers
    momentum_exhaustion = abs(data['momentum_second_deriv']) * data['divergence_persistence']
    
    # Dynamic factor generation
    factor = (short_term_div * volume_confirmation + 
              medium_term_div * volume_confirmation * 0.7 +
              data['volatility_volume_div'] * data['volume_asymmetry_ratio'] * 0.5 +
              momentum_asymmetry * volume_validation * 0.3 +
              momentum_exhaustion * data['divergence_persistence'] * 0.2)
    
    # Clean up intermediate columns
    cols_to_drop = ['up_day', 'down_day', 'up_day_volume_conc', 'down_day_volume_conc',
                   'positive_momentum_high_volume', 'negative_momentum_low_volume']
    data = data.drop(columns=[col for col in cols_to_drop if col in data.columns])
    
    return factor
