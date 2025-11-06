import pandas as pd
import numpy as np
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    """
    Adaptive Momentum-Volume Volatility Factor
    
    Economic intuition: Captures the interaction between momentum persistence, volume confirmation,
    and volatility stability across asymmetric horizons. The factor identifies stocks where price
    momentum is supported by increasing volume activity while volatility remains contained,
    suggesting sustainable price movements with strong market participation.
    
    Key innovations:
    - Asymmetric momentum horizons (5, 13, 34 days) for better market cycle capture
    - Volume-weighted momentum to emphasize high-conviction moves
    - Volatility containment ratio to filter noisy signals
    - Adaptive smoothing based on recent volatility regime
    - Dollar volume weighting for liquidity adjustment
    """
    
    horizons = [5, 13, 34]
    
    # Multi-horizon momentum components
    momentum_components = []
    volume_confirmation_components = []
    
    for horizon in horizons:
        # Volume-weighted momentum: price change amplified by volume expansion
        price_momentum = df['close'] / df['close'].shift(horizon) - 1
        volume_trend = df['volume'] / df['volume'].shift(horizon)
        volume_weighted_momentum = price_momentum * volume_trend
        momentum_components.append(volume_weighted_momentum)
        
        # Volume confirmation: current volume relative to recent average
        volume_ratio = df['volume'] / df['volume'].rolling(window=horizon).mean()
        volume_confirmation_components.append(volume_ratio)
    
    # Geometric combination of momentum across horizons
    momentum_geo = (momentum_components[0].abs() * 
                   momentum_components[1].abs() * 
                   momentum_components[2].abs()) ** (1/3)
    momentum_geo = momentum_geo * np.sign(momentum_components[1])  # Use medium-term direction
    
    # Volume confirmation score (arithmetic mean for additive effect)
    volume_confirmation = (volume_confirmation_components[0] + 
                          volume_confirmation_components[1] + 
                          volume_confirmation_components[2]) / 3
    
    # Volatility containment: measure of volatility stability
    daily_volatility = (df['high'] - df['low']) / df['close']
    vol_short_term = daily_volatility.rolling(window=5).std()
    vol_medium_term = daily_volatility.rolling(window=13).std()
    
    # Volatility containment ratio: current volatility relative to recent ranges
    volatility_containment = daily_volatility / (vol_short_term + vol_medium_term + 1e-7)
    
    # Core factor: Momentum amplified by volume confirmation, filtered by volatility containment
    raw_factor = momentum_geo * volume_confirmation / (volatility_containment + 1e-7)
    
    # Adaptive smoothing based on recent volatility regime
    recent_volatility = daily_volatility.rolling(window=8).mean()
    smooth_window = np.where(recent_volatility > recent_volatility.median(), 5, 13)
    
    # Apply adaptive smoothing
    smoothed_factor = raw_factor.copy()
    for i in range(len(raw_factor)):
        if i >= max(smooth_window):
            window = int(smooth_window[i])
            smoothed_factor.iloc[i] = raw_factor.iloc[i-window+1:i+1].mean()
    
    # Dollar volume weighting for liquidity adjustment
    dollar_volume = df['close'] * df['volume']
    dollar_weight = dollar_volume / dollar_volume.rolling(window=21).mean()
    
    final_factor = smoothed_factor * dollar_weight
    
    return final_factor
