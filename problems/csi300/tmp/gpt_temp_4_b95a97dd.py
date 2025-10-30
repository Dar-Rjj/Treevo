import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Acceleration with Volume-Price Convergence
    # Calculate 5-day price momentum
    momentum_5d = data['close'] / data['close'].shift(5) - 1
    
    # Calculate 3-day price momentum
    momentum_3d = data['close'] / data['close'].shift(3) - 1
    
    # Compute 3-day momentum acceleration
    momentum_acceleration = (momentum_5d - momentum_3d) / (momentum_3d + 1e-8)
    
    # Calculate VWAP_5d (Volume Weighted Average Price over 5 days)
    vwap_5d = (data['amount'].rolling(window=5).sum() / data['volume'].rolling(window=5).sum()).replace([np.inf, -np.inf], np.nan)
    
    # Calculate volume-weighted price convergence
    volume_convergence = (data['close'] - vwap_5d) / (vwap_5d + 1e-8)
    
    # Combine momentum acceleration and volume convergence
    factor1 = momentum_acceleration * volume_convergence
    
    # High-Low Breakout Efficiency
    # Calculate True Range
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Compute breakout efficiency
    breakout_efficiency = (data['close'] - data['low']) / (true_range + 1e-8)
    
    # Calculate volume confirmation
    volume_5d_avg = data['volume'].rolling(window=5).mean()
    volume_confirmation = np.sign(data['close'] - data['open']) * (data['volume'] / (volume_5d_avg + 1e-8))
    
    # Combine breakout efficiency and volume confirmation
    factor2 = breakout_efficiency * volume_confirmation
    
    # Price Impact Efficiency
    # Calculate Dollar Flow
    dollar_flow = data['amount'] * np.sign(data['close'] - data['open'])
    
    # Compute price impact efficiency
    price_impact_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Combine price impact efficiency and dollar flow
    factor3 = price_impact_efficiency / (abs(dollar_flow) + 1e-8)
    
    # Volatility-Adaptive Momentum
    # Calculate 10-day price volatility
    volatility_10d = data['close'].rolling(window=10).std()
    
    # Compute 5-day momentum
    momentum_5d_vol = data['close'] / data['close'].shift(5) - 1
    
    # Identify volatility regime (high_vol vs low_vol based on rolling percentile)
    vol_percentile = volatility_10d.rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
    volatility_regime = np.where(vol_percentile > 0.7, 1.0, 0.5)  # High vol = 1.0, Low vol = 0.5
    
    # Adaptive combination
    factor4 = momentum_5d_vol * volatility_regime
    
    # Intraday Gap Recovery
    # Calculate opening gap magnitude
    gap_magnitude = abs(data['open'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    
    # Compute gap recovery ratio
    gap_recovery_ratio = (data['close'] - data['open']) / (abs(data['open'] - data['close'].shift(1)) + 1e-8)
    
    # Combine gap magnitude and recovery ratio
    factor5 = gap_magnitude * gap_recovery_ratio
    
    # Final factor combination (equal weighted)
    final_factor = (factor1 + factor2 + factor3 + factor4 + factor5) / 5
    
    return final_factor
