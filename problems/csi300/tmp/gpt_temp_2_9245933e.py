import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Compression Dynamics
    # Multi-Timeframe Volatility Ratios
    data['short_term_vol'] = data['close'].rolling(window=5).std()
    data['medium_term_vol'] = data['close'].rolling(window=10).std()
    data['long_term_vol'] = data['close'].rolling(window=10).std().shift(10)
    
    # Short-term Compression
    data['short_term_compression'] = data['short_term_vol'] / data['short_term_vol'].shift(5)
    
    # Medium-term Compression
    data['medium_term_compression'] = data['medium_term_vol'] / data['long_term_vol']
    
    # Volatility Momentum
    data['compression_acceleration'] = data['short_term_compression'] - data['short_term_compression'].shift(5)
    data['volatility_regime_multiplier'] = np.sign(data['compression_acceleration']) * np.abs(data['compression_acceleration'])
    
    # Price-Volume Asymmetry Structure
    # Calculate daily returns for volume efficiency
    data['close_ret'] = data['close'].pct_change()
    
    # Directional Volume Efficiency
    upside_volume = []
    downside_volume = []
    price_ranges = []
    
    for i in range(len(data)):
        if i < 5:
            upside_volume.append(np.nan)
            downside_volume.append(np.nan)
            price_ranges.append(np.nan)
            continue
            
        upside_vol = 0
        downside_vol = 0
        total_range = 0
        
        for j in range(5):
            idx = i - j
            if data['close_ret'].iloc[idx] > 0:
                upside_vol += data['volume'].iloc[idx]
            elif data['close_ret'].iloc[idx] < 0:
                downside_vol += data['volume'].iloc[idx]
            total_range += data['high'].iloc[idx] - data['low'].iloc[idx]
        
        upside_volume.append(upside_vol)
        downside_volume.append(downside_vol)
        price_ranges.append(total_range)
    
    data['upside_volume'] = upside_volume
    data['downside_volume'] = downside_volume
    data['total_range_5d'] = price_ranges
    
    data['upside_volume_efficiency'] = data['upside_volume'] / data['total_range_5d']
    data['downside_volume_efficiency'] = data['downside_volume'] / data['total_range_5d']
    
    # Asymmetry Multipliers
    data['efficiency_ratio'] = data['upside_volume_efficiency'] / data['downside_volume_efficiency']
    data['volume_autocorrelation'] = data['volume'].rolling(window=5).corr(data['volume'].shift(5))
    
    # Liquidity Dynamics Integration
    # Price Impact Analysis
    data['unit_price_impact'] = (data['high'] - data['low']) / data['amount']
    data['impact_trend'] = data['unit_price_impact'] / data['unit_price_impact'].shift(1)
    
    # Amount-Volume Efficiency
    data['amount_per_volume'] = data['amount'] / data['volume']
    data['amount_trend_multiplier'] = data['amount'] / data['amount'].rolling(window=5).mean()
    
    # Bidirectional Pressure Synchronization
    # Order Flow Imbalance
    up_tick_pressure = []
    down_tick_pressure = []
    
    for i in range(len(data)):
        if i < 5:
            up_tick_pressure.append(np.nan)
            down_tick_pressure.append(np.nan)
            continue
            
        up_pressure = 0
        down_pressure = 0
        
        for j in range(5):
            idx = i - j
            if data['close_ret'].iloc[idx] > 0:
                up_pressure += data['volume'].iloc[idx]
            elif data['close_ret'].iloc[idx] < 0:
                down_pressure += data['volume'].iloc[idx]
        
        up_tick_pressure.append(up_pressure)
        down_tick_pressure.append(down_pressure)
    
    data['up_tick_pressure'] = up_tick_pressure
    data['down_tick_pressure'] = down_tick_pressure
    
    # Pressure Differential
    data['relative_buy_pressure'] = data['up_tick_pressure'] / (data['up_tick_pressure'] + data['down_tick_pressure'])
    data['pressure_momentum'] = data['relative_buy_pressure'] - data['relative_buy_pressure'].shift(1)
    
    # Fractal Efficiency Enhancement
    # Price Path Complexity
    data['range_efficiency'] = (data['close'] - data['close'].shift(4)) / data['total_range_5d']
    data['directional_consistency'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['close'].shift(1) - data['close'].shift(2))
    
    # Volume Distribution Pattern
    data['volume_clustering'] = data['volume'].rolling(window=5).max() / data['volume'].rolling(window=5).min()
    data['volume_price_correlation'] = data['close'].rolling(window=5).corr(data['volume'])
    
    # Composite Synchronization Core
    # Volatility-Asymmetry-Liquidity Alignment
    data['triple_synchronization'] = data['short_term_compression'] * data['efficiency_ratio'] * data['impact_trend']
    data['bidirectional_alignment'] = data['medium_term_compression'] * data['volume_autocorrelation'] * data['amount_trend_multiplier']
    
    # Efficiency-Weighted Factor
    data['range_efficiency_weighted'] = data['triple_synchronization'] * data['range_efficiency']
    data['volume_clustering_adjusted'] = data['bidirectional_alignment'] / data['volume_clustering']
    
    # Base factor combination
    data['base_factor'] = data['range_efficiency_weighted'] + data['volume_clustering_adjusted']
    
    # Multi-Scale Signal Refinement
    # Volatility Regime Adaptation
    data['high_compression_scaled'] = data['base_factor'] * data['volatility_regime_multiplier']
    data['low_compression_enhanced'] = data['base_factor'] / data['volatility_regime_multiplier']
    
    # Final factor with regime adaptation
    data['regime_adapted_factor'] = np.where(
        np.abs(data['volatility_regime_multiplier']) > 1,
        data['high_compression_scaled'],
        data['low_compression_enhanced']
    )
    
    # Trend Phase Confirmation
    data['pressure_momentum_aligned'] = data['regime_adapted_factor'] * data['pressure_momentum']
    data['volume_breakout_ratio'] = data['volume'] / data['volume'].rolling(window=5).mean().shift(5)
    data['volume_breakout_confirmed'] = data['regime_adapted_factor'] * data['volume_breakout_ratio']
    
    # Final composite factor
    data['final_factor'] = (data['pressure_momentum_aligned'] + data['volume_breakout_confirmed']) / 2
    
    # Return the final factor series
    return data['final_factor']
