import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multi-timeframe momentum, volatility regime classification,
    intraday price action, volume regime analysis, and price level context integration.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Dynamic Timeframe Momentum Analysis
    data['price_momentum_2d'] = data['close'] / data['close'].shift(2)
    data['volume_momentum_2d'] = data['volume'] / data['volume'].shift(2)
    data['divergence_2d'] = data['price_momentum_2d'] - data['volume_momentum_2d']
    
    data['price_momentum_5d'] = data['close'] / data['close'].shift(5)
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5)
    data['divergence_5d'] = data['price_momentum_5d'] - data['volume_momentum_5d']
    
    data['price_momentum_15d'] = data['close'] / data['close'].shift(15)
    data['volume_momentum_15d'] = data['volume'] / data['volume'].shift(15)
    data['divergence_15d'] = data['price_momentum_15d'] - data['volume_momentum_15d']
    
    data['price_momentum_40d'] = data['close'] / data['close'].shift(40)
    data['volume_momentum_40d'] = data['volume'] / data['volume'].shift(40)
    data['divergence_40d'] = data['price_momentum_40d'] - data['volume_momentum_40d']
    
    # Refined Volatility Regime Classification
    data['returns'] = data['close'].pct_change()
    data['ultra_short_vol'] = data['returns'].rolling(window=2).std()
    data['short_term_vol'] = data['returns'].rolling(window=5).std()
    data['medium_term_vol'] = data['returns'].rolling(window=15).std()
    
    data['vol_regime_score'] = (data['ultra_short_vol'] / data['short_term_vol']) + (data['short_term_vol'] / data['medium_term_vol'])
    
    # Dynamic Timeframe Weighting based on Volatility Regime
    def get_weights(score):
        if score > 2.5:
            return [0.5, 0.3, 0.15, 0.05]  # High volatility
        elif score >= 1.5:
            return [0.25, 0.35, 0.25, 0.15]  # Moderate volatility
        else:
            return [0.1, 0.2, 0.35, 0.35]  # Low volatility
    
    weights = data['vol_regime_score'].apply(get_weights)
    data['ultra_short_weight'] = weights.apply(lambda x: x[0])
    data['short_term_weight'] = weights.apply(lambda x: x[1])
    data['medium_term_weight'] = weights.apply(lambda x: x[2])
    data['long_term_weight'] = weights.apply(lambda x: x[3])
    
    # Enhanced Intraday Price Action Analysis
    data['normalized_range'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['close_to_open_ratio'] = data['close'] / data['open']
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_momentum'] = data['intraday_momentum'].replace([np.inf, -np.inf], np.nan)
    
    data['gap_size'] = abs(data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_fill_indicator'] = (data['close'] - data['open']) / (data['open'] - data['close'].shift(1))
    data['gap_fill_indicator'] = data['gap_fill_indicator'].replace([np.inf, -np.inf], np.nan)
    data['gap_direction_persistence'] = np.sign(data['close'] - data['open']) * np.sign(data['open'] - data['close'].shift(1))
    
    data['range_adjusted_momentum'] = data['intraday_momentum'] * (1 + data['normalized_range'])
    data['gap_adjusted_factor'] = data['range_adjusted_momentum'] * (1 + data['gap_size'] * data['gap_fill_indicator'])
    data['intraday_score'] = data['gap_adjusted_factor'] * (1 + 0.2 * data['gap_direction_persistence'])
    
    # Volume Regime Analysis
    data['volume_percentile'] = data['volume'].rolling(window=30).apply(lambda x: (x.iloc[-1] > x).mean(), raw=False)
    data['volume_multiplier'] = np.where(data['volume_percentile'] > 0.8, 1.8,
                                       np.where(data['volume_percentile'] < 0.2, 0.6, 1.0))
    
    # Price Level Context Integration
    data['high_30d'] = data['high'].rolling(window=30).max()
    data['low_30d'] = data['low'].rolling(window=30).min()
    data['current_position'] = (data['close'] - data['low_30d']) / (data['high_30d'] - data['low_30d'])
    
    prev_low_30d = data['low_30d'].shift(5)
    prev_high_30d = data['high_30d'].shift(5)
    data['position_momentum'] = data['current_position'] - ((data['close'].shift(5) - prev_low_30d) / (prev_high_30d - prev_low_30d))
    
    def get_position_multiplier(position, momentum):
        if position > 0.9 and momentum > 0:
            return 1.5
        elif position < 0.1 and momentum < 0:
            return 1.3
        elif position > 0.8:
            return 0.7
        elif position < 0.2:
            return 0.8
        else:
            return 1.0
    
    data['position_multiplier'] = data.apply(
        lambda row: get_position_multiplier(row['current_position'], row['position_momentum']), axis=1
    )
    
    # Final Alpha Construction
    data['weighted_divergence'] = (
        data['divergence_2d'] * data['ultra_short_weight'] +
        data['divergence_5d'] * data['short_term_weight'] +
        data['divergence_15d'] * data['medium_term_weight'] +
        data['divergence_40d'] * data['long_term_weight']
    )
    
    data['volume_enhanced'] = data['weighted_divergence'] * data['volume_multiplier']
    data['intraday_integrated'] = data['volume_enhanced'] * data['intraday_score']
    data['final_alpha'] = data['intraday_integrated'] * data['position_multiplier']
    
    return data['final_alpha']
