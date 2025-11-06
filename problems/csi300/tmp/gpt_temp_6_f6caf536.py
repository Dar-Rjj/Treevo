import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic price and volume changes
    data['price_change'] = data['close'] / data['close'].shift(1) - 1
    data['volume_change'] = data['volume'] / data['volume'].shift(1)
    
    # 1. Directional Volume-Weighted Efficiency
    # Up-Day and Down-Day Volume Efficiency
    up_mask = data['close'] > data['close'].shift(1)
    down_mask = data['close'] < data['close'].shift(1)
    
    data['up_volume_efficiency'] = np.where(up_mask, data['price_change'] * data['volume_change'], 0)
    data['down_volume_efficiency'] = np.where(down_mask, abs(data['price_change']) * data['volume_change'], 0)
    
    # Directional Volume Ratio with smoothing
    vol_ratio = (data['up_volume_efficiency'].rolling(3, min_periods=1).mean() + 1e-8) / \
                (data['down_volume_efficiency'].rolling(3, min_periods=1).mean() + 1e-8)
    data['directional_volume_ratio'] = np.log(vol_ratio)
    
    # Multi-timeframe directional efficiency
    for window in [3, 10, 20]:
        up_count = data['close'].rolling(window).apply(lambda x: (x > x.shift(1)).sum(), raw=False)
        down_count = data['close'].rolling(window).apply(lambda x: (x < x.shift(1)).sum(), raw=False)
        data[f'directional_efficiency_{window}d'] = (up_count - down_count) / window
    
    # Efficiency momentum differences
    data['efficiency_momentum_short_med'] = data['directional_efficiency_3d'] - data['directional_efficiency_10d']
    data['efficiency_momentum_med_long'] = data['directional_efficiency_10d'] - data['directional_efficiency_20d']
    data['efficiency_momentum_short_long'] = data['directional_efficiency_3d'] - data['directional_efficiency_20d']
    
    # Volume-weighted efficiency momentum
    efficiency_momentum = (data['efficiency_momentum_short_med'] + 
                          data['efficiency_momentum_med_long'] + 
                          data['efficiency_momentum_short_long']) / 3
    data['volume_efficiency_momentum'] = efficiency_momentum * data['directional_volume_ratio']
    
    # 2. Price-Volume Divergence Patterns
    # Volume-Price direction mismatch
    up_price_decr_vol = (data['close'] > data['close'].shift(1)) & (data['volume'] < data['volume'].shift(1))
    down_price_incr_vol = (data['close'] < data['close'].shift(1)) & (data['volume'] > data['volume'].shift(1))
    
    data['divergence_signal'] = 0
    data.loc[up_price_decr_vol, 'divergence_signal'] = -1
    data.loc[down_price_incr_vol, 'divergence_signal'] = -1
    
    # Price-Volume consistency patterns
    up_price_incr_vol = (data['close'] > data['close'].shift(1)) & (data['volume'] > data['volume'].shift(1))
    down_price_decr_vol = (data['close'] < data['close'].shift(1)) & (data['volume'] < data['volume'].shift(1))
    data.loc[up_price_incr_vol, 'divergence_signal'] = 1
    data.loc[down_price_decr_vol, 'divergence_signal'] = 1
    
    # Multi-timeframe divergence intensity
    for window in [3, 10, 20]:
        divergence_count = data['divergence_signal'].rolling(window).apply(
            lambda x: (x == -1).sum() if len(x) == window else 0, raw=False
        )
        price_movement = data['close'].pct_change(window).abs()
        data[f'divergence_intensity_{window}d'] = divergence_count * price_movement
    
    # Divergence momentum
    data['divergence_momentum_3_10'] = data['divergence_intensity_3d'] - data['divergence_intensity_10d']
    data['divergence_momentum_10_20'] = data['divergence_intensity_10d'] - data['divergence_intensity_20d']
    
    divergence_momentum = (data['divergence_momentum_3_10'] + data['divergence_momentum_10_20']) / 2
    data['directional_divergence_score'] = divergence_momentum * data['divergence_signal']
    
    # 3. Gap-Volume Efficiency Analysis
    # Gap metrics
    data['opening_gap'] = (data['open'] / data['close'].shift(1) - 1)
    data['gap_fill_efficiency'] = (data['high'] - data['low']) / abs(data['opening_gap'] + 1e-8)
    
    # Volume concentration around gaps
    gap_volume_ratio = data['volume'] / data['volume'].rolling(5).mean()
    data['gap_volume_concentration'] = gap_volume_ratio * abs(data['opening_gap'])
    
    # Gap persistence
    gap_direction = np.sign(data['opening_gap'])
    data['gap_persistence'] = gap_direction.rolling(3).apply(
        lambda x: (x == x.iloc[0]).sum() if len(x) == 3 else 0, raw=False
    )
    
    # Gap efficiency score
    data['gap_efficiency_score'] = (data['gap_fill_efficiency'] * 
                                   data['gap_volume_concentration'] * 
                                   data['gap_persistence'])
    
    # 4. Price-Range Volume Concentration Analysis
    # Volume distribution efficiency
    price_range = data['high'] - data['low']
    data['volume_per_price_unit'] = data['volume'] / (price_range + 1e-8)
    data['volume_concentration_ratio'] = data['volume_per_price_unit'] / data['volume_per_price_unit'].rolling(10).mean()
    
    # Range-volume momentum
    range_change = price_range / price_range.shift(1)
    data['range_volume_momentum'] = range_change * data['volume_concentration_ratio']
    
    # Range stability score
    range_std = price_range.rolling(10).std()
    data['range_stability_score'] = price_range / (range_std + 1e-8)
    
    # Volume-concentrated range signals
    data['range_volume_signal'] = data['range_volume_momentum'] * data['range_stability_score'] * data['volume_concentration_ratio']
    
    # 5. Final Alpha Factor Synthesis
    # Combine all components with appropriate weighting
    efficiency_component = data['volume_efficiency_momentum'].fillna(0)
    divergence_component = -data['directional_divergence_score'].fillna(0)  # Negative divergence is bullish
    gap_component = data['gap_efficiency_score'].fillna(0)
    range_component = data['range_volume_signal'].fillna(0)
    
    # Multi-timeframe synthesis
    short_term = (efficiency_component.rolling(3).mean() + 
                 divergence_component.rolling(3).mean() + 
                 gap_component.rolling(3).mean() + 
                 range_component.rolling(3).mean()) / 4
    
    medium_term = (efficiency_component.rolling(10).mean() + 
                  divergence_component.rolling(10).mean() + 
                  gap_component.rolling(10).mean() + 
                  range_component.rolling(10).mean()) / 4
    
    long_term = (efficiency_component.rolling(20).mean() + 
                divergence_component.rolling(20).mean() + 
                gap_component.rolling(20).mean() + 
                range_component.rolling(20).mean()) / 4
    
    # Final asymmetric momentum-efficiency factor
    alpha_factor = (0.4 * short_term + 0.35 * medium_term + 0.25 * long_term)
    
    # Normalize and return
    alpha_factor = (alpha_factor - alpha_factor.rolling(50).mean()) / (alpha_factor.rolling(50).std() + 1e-8)
    
    return alpha_factor
