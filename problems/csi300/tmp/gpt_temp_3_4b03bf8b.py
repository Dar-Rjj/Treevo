import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor components
    factors = pd.DataFrame(index=data.index)
    
    # 1. Asymmetric Volatility Momentum
    # Calculate multi-timeframe returns
    returns_3d = data['close'].pct_change(3)
    returns_10d = data['close'].pct_change(10)
    returns_20d = data['close'].pct_change(20)
    
    # Compute volatility asymmetry
    upside_vol = (data['high'] - data['close']) / data['close']
    downside_vol = (data['close'] - data['low']) / data['close']
    vol_asymmetry = upside_vol / (downside_vol + 1e-8)
    
    # Generate momentum factors
    factors['vol_momentum_3d'] = returns_3d * vol_asymmetry
    factors['vol_momentum_10d'] = returns_10d * vol_asymmetry
    factors['vol_momentum_20d'] = returns_20d * vol_asymmetry
    
    # 2. Volume-Weighted Range Breakout
    # Calculate price-volume flow
    price_volume_flow = (data['close'] - data['close'].shift(1)) * data['volume']
    
    # Compute cumulative flow acceleration (5-day rolling)
    flow_acceleration = price_volume_flow.rolling(window=5).mean() - price_volume_flow.rolling(window=10).mean()
    
    # Calculate range breakout (current range vs 10-day average range)
    daily_range = (data['high'] - data['low']) / data['close']
    avg_range = daily_range.rolling(window=10).mean()
    range_breakout = daily_range / (avg_range + 1e-8)
    
    factors['volume_breakout'] = range_breakout * flow_acceleration
    
    # 3. Intraday Gap Efficiency
    # Calculate gap strength
    gap_strength = abs(data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    
    # Compute gap direction alignment (1 if gap direction matches intraday move, -1 otherwise)
    gap_direction = np.sign(data['open'] - data['close'].shift(1))
    intraday_direction = np.sign(data['close'] - data['open'])
    gap_alignment = gap_direction * intraday_direction
    
    # Calculate intraday pattern strength
    intraday_range_utilization = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    factors['gap_efficiency'] = gap_strength * gap_alignment * intraday_range_utilization
    
    # 4. Volume-Asymmetry Gap Reversal
    # Calculate opening gap percentage
    gap_percentage = (data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    
    # Compute volume asymmetry (upside vs downside volume)
    upside_volume = np.where(data['close'] > data['open'], data['volume'], 0)
    downside_volume = np.where(data['close'] < data['open'], data['volume'], 0)
    
    upside_volume_sum = pd.Series(upside_volume, index=data.index).rolling(window=5).sum()
    downside_volume_sum = pd.Series(downside_volume, index=data.index).rolling(window=5).sum()
    volume_asymmetry = upside_volume_sum / (downside_volume_sum + 1e-8)
    
    factors['gap_reversal'] = gap_percentage * volume_asymmetry
    
    # 5. Microstructure-Adaptive Order Flow
    # Calculate range utilization
    range_utilization = abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Compute microstructure efficiency (price impact efficiency)
    price_efficiency = abs(data['close'] - data['open']) / (data['volume'] + 1e-8)
    efficiency_score = price_efficiency.rolling(window=10).apply(
        lambda x: x.mean() / (x.std() + 1e-8) if x.std() > 0 else 0
    )
    
    # Calculate signed volume
    signed_volume = np.sign(data['close'] - data['open']) * data['volume']
    
    factors['microstructure_flow'] = signed_volume * efficiency_score * range_utilization
    
    # 6. Composite Alpha Generation
    # Normalize all factors
    normalized_factors = factors.apply(lambda x: (x - x.rolling(window=20).mean()) / (x.rolling(window=20).std() + 1e-8))
    
    # Dynamic regime weighting based on recent volatility
    recent_volatility = data['close'].pct_change().rolling(window=10).std()
    vol_regime = (recent_volatility > recent_volatility.rolling(window=20).mean()).astype(int)
    
    # High volatility regime: emphasize momentum and reversal factors
    # Low volatility regime: emphasize microstructure and efficiency factors
    momentum_weight = 0.4 + 0.2 * vol_regime
    breakout_weight = 0.2 + 0.1 * vol_regime
    efficiency_weight = 0.3 - 0.2 * vol_regime
    reversal_weight = 0.1 + 0.1 * vol_regime
    
    # Combine factors with dynamic weights
    composite_alpha = (
        momentum_weight * (normalized_factors['vol_momentum_3d'] + 
                          normalized_factors['vol_momentum_10d'] + 
                          normalized_factors['vol_momentum_20d']) / 3 +
        breakout_weight * normalized_factors['volume_breakout'] +
        efficiency_weight * normalized_factors['gap_efficiency'] +
        reversal_weight * normalized_factors['gap_reversal'] +
        efficiency_weight * normalized_factors['microstructure_flow']
    )
    
    # Final normalization
    final_factor = (composite_alpha - composite_alpha.rolling(window=20).mean()) / (composite_alpha.rolling(window=20).std() + 1e-8)
    
    return final_factor
