import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining volatility-scaled momentum, volume-price divergence,
    geometric momentum, and intraday momentum convergence.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Scaled Multi-Timeframe Momentum
    # Calculate momentum for different timeframes
    mom_3d = data['close'] / data['close'].shift(3) - 1
    mom_5d = data['close'] / data['close'].shift(5) - 1
    mom_10d = data['close'] / data['close'].shift(10) - 1
    
    # Calculate corresponding volatilities
    vol_3d = data['close'].pct_change().rolling(window=3).std()
    vol_5d = data['close'].pct_change().rolling(window=5).std()
    vol_10d = data['close'].pct_change().rolling(window=10).std()
    
    # Volatility-scaled momentum (avoid division by zero)
    mom_3d_scaled = mom_3d / (vol_3d + 1e-8)
    mom_5d_scaled = mom_5d / (vol_5d + 1e-8)
    mom_10d_scaled = mom_10d / (vol_10d + 1e-8)
    
    # Multi-timeframe convergence using geometric mean
    scaled_momentums = pd.concat([mom_3d_scaled, mom_5d_scaled, mom_10d_scaled], axis=1)
    momentum_factor = scaled_momentums.apply(lambda x: np.sign(x).prod() * (abs(x).prod())**(1/3), axis=1)
    
    # Volume-Price Divergence with Intraday Confirmation
    # Core Price Strength
    intraday_position = (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)
    opening_strength = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    price_efficiency = (data['close'] - data['close'].shift(1)) / (data['high'] - data['low'] + 1e-8)
    
    # Volume Divergence Analysis
    volume_accel = data['volume'] / (data['volume'].shift(1) + 1e-8)
    volume_trend = data['volume'] / data['volume'].rolling(window=5).mean()
    volume_vol = data['volume'] / data['volume'].rolling(window=10).mean()
    
    # Intraday Confirmation (simplified)
    morning_confirmation = (data['high'] - data['open']) / (data['open'] - data['low'] + 1e-8)
    afternoon_confirmation = (data['close'] - data['low']) / (data['high'] - data['close'] + 1e-8)
    session_alignment = np.sign(morning_confirmation * afternoon_confirmation)
    
    # Combined Volume-Price Factor
    price_components = pd.concat([intraday_position, opening_strength, price_efficiency], axis=1)
    volume_components = pd.concat([volume_accel, volume_trend, volume_vol], axis=1)
    
    price_score = price_components.apply(lambda x: x.mean(), axis=1)
    volume_score = volume_components.apply(lambda x: x.mean(), axis=1)
    volume_price_factor = price_score * volume_score * session_alignment
    
    # Multi-Timeframe Geometric Momentum
    # Short-term geometric returns
    geo_1d = data['close'] / data['close'].shift(1) - 1
    geo_2d = (data['close'] / data['close'].shift(2))**(1/2) - 1
    geo_3d = (data['close'] / data['close'].shift(3))**(1/3) - 1
    
    # Medium-term geometric returns
    geo_5d = (data['close'] / data['close'].shift(5))**(1/5) - 1
    geo_7d = (data['close'] / data['close'].shift(7))**(1/7) - 1
    geo_10d = (data['close'] / data['close'].shift(10))**(1/10) - 1
    
    # Volatility-adjusted geometric returns
    short_term_vol = data['close'].pct_change().rolling(window=3).std()
    medium_term_vol = data['close'].pct_change().rolling(window=10).std()
    
    short_geo_adj = pd.concat([geo_1d, geo_2d, geo_3d], axis=1).apply(lambda x: x.mean(), axis=1) / (short_term_vol + 1e-8)
    medium_geo_adj = pd.concat([geo_5d, geo_7d, geo_10d], axis=1).apply(lambda x: x.mean(), axis=1) / (medium_term_vol + 1e-8)
    
    geometric_factor = np.sign(short_geo_adj * medium_geo_adj) * (abs(short_geo_adj * medium_geo_adj))**(1/2)
    
    # Convergent Intraday Momentum Factor
    # Opening Session Analysis
    gap_strength = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    morning_momentum = (data['high'] - data['open']) / (data['open'] + 1e-8)
    morning_support = (data['open'] - data['low']) / (data['open'] + 1e-8)
    
    # Closing Session Analysis
    afternoon_momentum = (data['close'] - data['high']) / (data['high'] + 1e-8)
    closing_strength = (data['close'] - data['low']) / (data['low'] + 1e-8)
    session_persistence = (data['close'] - data['open']) / (data['open'] + 1e-8)
    
    # Volume confirmation (simplified using daily volume)
    morning_volume_intensity = data['volume'] / data['volume'].rolling(window=5).mean()
    afternoon_volume_intensity = data['volume'] / data['volume'].rolling(window=5).mean()
    volume_divergence = morning_volume_intensity / (afternoon_volume_intensity + 1e-8)
    
    # Combined Intraday Factor
    opening_components = pd.concat([gap_strength, morning_momentum, morning_support], axis=1)
    closing_components = pd.concat([afternoon_momentum, closing_strength, session_persistence], axis=1)
    
    opening_score = opening_components.apply(lambda x: x.mean(), axis=1)
    closing_score = closing_components.apply(lambda x: x.mean(), axis=1)
    intraday_factor = opening_score * closing_score * volume_divergence
    
    # Final Combined Alpha Factor
    # Geometric mean of all four components
    factors = pd.concat([momentum_factor, volume_price_factor, geometric_factor, intraday_factor], axis=1)
    final_factor = factors.apply(lambda x: np.sign(x).prod() * (abs(x).prod())**(1/4), axis=1)
    
    return final_factor
