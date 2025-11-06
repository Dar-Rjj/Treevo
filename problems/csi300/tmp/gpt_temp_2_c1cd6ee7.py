import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Gap-Range Analysis
    data['gap_pct'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Range Efficiency
    close_open_diff = data['close'] - data['open']
    data['range_efficiency'] = 1 / (1 + np.abs((data['high'] - data['low']) / np.where(close_open_diff != 0, close_open_diff, 1) - 1))
    
    # Efficiency-Weighted Intraday
    data['eff_weighted_intraday'] = ((data['close'] - data['open']) / data['open']) * data['range_efficiency']
    
    # Momentum Acceleration with Range Divergence
    data['momentum_5d'] = data['close'] / data['close'].shift(5)
    data['momentum_20d'] = data['close'] / data['close'].shift(20)
    data['momentum_acceleration'] = (data['momentum_5d'] - data['momentum_20d']) / np.abs(data['momentum_5d'])
    
    # Calculate average range on up and down days
    up_days = data['close'] > data['close'].shift(1)
    down_days = data['close'] < data['close'].shift(1)
    
    # Rolling range calculations
    data['daily_range'] = data['high'] - data['low']
    
    # Average range on up days (t-4 to t)
    up_range_avg = data['daily_range'].rolling(window=5).apply(
        lambda x: x[up_days.loc[x.index].fillna(False)].mean() if up_days.loc[x.index].fillna(False).any() else 1, 
        raw=False
    )
    
    # Average range on down days (t-4 to t)
    down_range_avg = data['daily_range'].rolling(window=5).apply(
        lambda x: x[down_days.loc[x.index].fillna(False)].mean() if down_days.loc[x.index].fillna(False).any() else 1, 
        raw=False
    )
    
    data['range_expansion'] = data['daily_range'] / up_range_avg
    data['range_compression'] = data['daily_range'] / down_range_avg
    
    # Volume-Amount Dynamics
    data['volume_acceleration'] = (data['volume'] - data['volume'].shift(5)) / data['volume'].shift(5)
    
    # Liquidity Efficiency
    data['liquidity_efficiency'] = data['amount'] / np.where(data['daily_range'] != 0, data['daily_range'], 1)
    
    # Order Flow Imbalance
    data['amount_change'] = (data['amount'] - data['amount'].shift(1)) / data['amount'].shift(1)
    data['order_flow_imbalance'] = data['amount_change'] - data['range_expansion']
    
    # Volatility-Liquidity Regime Assessment
    data['volatility_ratio'] = data['daily_range'] / data['daily_range'].rolling(window=10).mean()
    
    # Liquidity Regime Shift
    data['liquidity_regime_shift'] = data['liquidity_efficiency'] / data['liquidity_efficiency'].shift(5)
    
    # Microstructure Quality
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    data['microstructure_quality'] = ((data['high'] - data['low']) / np.where(close_open_diff != 0, close_open_diff, 1)) * \
                                   (np.abs(data['close'] - typical_price) / typical_price)
    
    # Cross-Dimensional Divergence Integration
    data['bullish_divergence'] = -data['momentum_5d'] * data['range_expansion']
    data['bearish_divergence'] = data['momentum_5d'] * data['range_compression']
    
    # Cross-Timeframe Alignment
    data['cross_timeframe_alignment'] = np.sign(data['eff_weighted_intraday']) * np.sign(data['amount_change'])
    
    # Composite Factor Generation
    # Base Momentum with regime adaptation
    volatility_weight = 1 / (1 + np.abs(data['volatility_ratio'] - 1))
    liquidity_weight = 1 / (1 + np.abs(data['liquidity_regime_shift'] - 1))
    
    data['base_momentum'] = data['momentum_acceleration'] * volatility_weight * liquidity_weight
    
    # Divergence Multiplier
    divergence_strength = (data['bullish_divergence'] - data['bearish_divergence']) * data['cross_timeframe_alignment']
    data['divergence_multiplier'] = 1 + np.tanh(divergence_strength)
    
    # Final Assembly with regime-dependent scaling
    regime_scale = np.where(data['volatility_ratio'] > 1.2, 0.8, 
                           np.where(data['volatility_ratio'] < 0.8, 1.2, 1.0))
    
    # Combine components multiplicatively
    data['composite_factor'] = (data['base_momentum'] * 
                               data['divergence_multiplier'] * 
                               data['order_flow_imbalance'] * 
                               data['microstructure_quality'] * 
                               regime_scale)
    
    return data['composite_factor']
