import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining multiple volume-price efficiency metrics
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # 1. Asymmetric Volume-Price Efficiency Factor
    # Calculate directional price efficiency
    upside_efficiency = np.where(data['close'] > data['open'], 
                                (data['close'] - data['open']) / data['open'], 0)
    downside_efficiency = np.where(data['close'] < data['open'], 
                                  (data['open'] - data['close']) / data['open'], 0)
    
    # Compute volume asymmetry
    upside_volume = np.where(data['close'] > data['open'], data['volume'], 0)
    downside_volume = np.where(data['close'] < data['open'], data['volume'], 0)
    
    # Rolling averages for volume asymmetry
    upside_volume_ma = upside_volume.rolling(window=5, min_periods=3).mean()
    downside_volume_ma = downside_volume.rolling(window=5, min_periods=3).mean()
    
    # Efficiency gap calculation
    efficiency_ratio = (upside_efficiency.rolling(window=5, min_periods=3).mean() + 1e-8) / \
                      (downside_efficiency.rolling(window=5, min_periods=3).mean() + 1e-8)
    volume_asymmetry = (upside_volume_ma + 1e-8) / (downside_volume_ma + 1e-8)
    
    asymmetric_factor = efficiency_ratio / (volume_asymmetry + 1e-8)
    
    # 2. Overnight Gap Momentum Persistence
    # Calculate overnight return
    overnight_return = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    overnight_gap_magnitude = np.abs(overnight_return)
    
    # Intraday momentum
    intraday_return = (data['close'] - data['open']) / (data['open'] + 1e-8)
    
    # Gap persistence measure
    gap_persistence = np.sign(overnight_return) * np.sign(intraday_return)
    gap_strength = overnight_gap_magnitude * gap_persistence
    
    # Volume confirmation
    gap_volume_ratio = data['volume'] / (data['volume'].rolling(window=10, min_periods=5).mean() + 1e-8)
    gap_factor = gap_strength * gap_volume_ratio
    
    # 3. Volume-Weighted Price Acceleration
    # Volume-weighted returns
    vwap_3d = (data['close'] * data['volume']).rolling(window=3, min_periods=2).sum() / \
              (data['volume'].rolling(window=3, min_periods=2).sum() + 1e-8)
    vwap_8d = (data['close'] * data['volume']).rolling(window=8, min_periods=5).sum() / \
              (data['volume'].rolling(window=8, min_periods=5).sum() + 1e-8)
    
    short_term_return = (vwap_3d - vwap_3d.shift(3)) / (vwap_3d.shift(3) + 1e-8)
    medium_term_return = (vwap_8d - vwap_8d.shift(8)) / (vwap_8d.shift(8) + 1e-8)
    
    # Acceleration metrics
    acceleration = short_term_return - medium_term_return
    volume_acceleration = data['volume'].pct_change(periods=3).rolling(window=5, min_periods=3).mean()
    
    acceleration_factor = acceleration * volume_acceleration
    
    # 4. Price Range Efficiency Factor
    # Actual vs expected range
    actual_range = (data['high'] - data['low']) / (data['close'] + 1e-8)
    expected_range = actual_range.rolling(window=20, min_periods=10).mean()
    
    # Volume per unit range efficiency
    range_efficiency = data['volume'] / ((data['high'] - data['low']) + 1e-8)
    efficiency_ratio_range = range_efficiency / (range_efficiency.rolling(window=10, min_periods=5).mean() + 1e-8)
    
    # Range efficiency signal
    range_factor = (actual_range - expected_range) * efficiency_ratio_range
    
    # 5. Multi-Timeframe Volume Divergence
    # Volume trends
    volume_short = data['volume'].rolling(window=5, min_periods=3).mean()
    volume_long = data['volume'].rolling(window=20, min_periods=10).mean()
    
    # Price-volume efficiency ratios
    price_change_short = data['close'].pct_change(periods=3).rolling(window=5, min_periods=3).mean()
    price_change_long = data['close'].pct_change(periods=8).rolling(window=10, min_periods=5).mean()
    
    volume_efficiency_short = price_change_short / (volume_short.pct_change(periods=3).rolling(window=5, min_periods=3).mean() + 1e-8)
    volume_efficiency_long = price_change_long / (volume_long.pct_change(periods=8).rolling(window=10, min_periods=5).mean() + 1e-8)
    
    # Divergence detection
    volume_divergence = volume_efficiency_short - volume_efficiency_long
    price_volume_alignment = np.sign(price_change_short) * np.sign(volume_efficiency_short)
    
    divergence_factor = volume_divergence * price_volume_alignment
    
    # Combine all factors with equal weights
    combined_factor = (
        asymmetric_factor.fillna(0) * 0.2 +
        gap_factor.fillna(0) * 0.2 +
        acceleration_factor.fillna(0) * 0.2 +
        range_factor.fillna(0) * 0.2 +
        divergence_factor.fillna(0) * 0.2
    )
    
    # Final normalization
    final_factor = (combined_factor - combined_factor.rolling(window=20, min_periods=10).mean()) / \
                   (combined_factor.rolling(window=20, min_periods=10).std() + 1e-8)
    
    return final_factor
