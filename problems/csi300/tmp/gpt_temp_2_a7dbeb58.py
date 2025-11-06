import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Range-Momentum Volume Efficiency Alpha Factor
    Combines efficient range utilization with momentum persistence and volume alignment
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Efficient Range Utilization
    # Intraday Efficiency: Absolute price change / Daily range
    daily_range = data['high'] - data['low']
    daily_range = daily_range.replace(0, np.nan)  # Avoid division by zero
    intraday_efficiency = abs(data['close'] - data['open']) / daily_range
    
    # Directional Efficiency - separate up and down days
    up_days = data['close'] > data['open']
    down_days = data['close'] < data['open']
    
    # Up day efficiency: (close - open) / (high - low)
    up_efficiency = np.where(up_days, (data['close'] - data['open']) / daily_range, 0)
    # Down day efficiency: (open - close) / (high - low)  
    down_efficiency = np.where(down_days, (data['open'] - data['close']) / daily_range, 0)
    
    # Ratio of directional efficiency (up/down bias)
    directional_ratio = up_efficiency / (down_efficiency + 1e-8)
    
    # 2. Integrate Momentum Components
    # Multi-timeframe momentum (3, 5, 8 days)
    momentum_3d = data['close'].pct_change(periods=3)
    momentum_5d = data['close'].pct_change(periods=5)
    momentum_8d = data['close'].pct_change(periods=8)
    
    # Momentum convergence (agreement across timeframes)
    momentum_convergence = (momentum_3d * momentum_5d * momentum_8d) ** (1/3)
    
    # Momentum persistence - count consecutive same-sign momentum days
    momentum_sign = np.sign(momentum_3d)
    persistence_count = momentum_sign.groupby((momentum_sign != momentum_sign.shift(1)).cumsum()).cumcount() + 1
    momentum_persistence = persistence_count * abs(momentum_3d)
    
    # 3. Apply Volume-Weighted Adjustment
    # Volume efficiency: volume per unit price change
    price_change = abs(data['close'] - data['open'])
    volume_efficiency = np.where(price_change > 0, data['volume'] / price_change, 0)
    
    # Volume-momentum alignment (5-day rolling correlation)
    volume_returns = data['volume'].pct_change()
    price_returns = data['close'].pct_change()
    
    # Calculate rolling correlation
    corr_window = 5
    volume_momentum_corr = pd.Series(index=data.index, dtype=float)
    
    for i in range(corr_window, len(data)):
        window_volume = volume_returns.iloc[i-corr_window:i]
        window_price = price_returns.iloc[i-corr_window:i]
        if len(window_volume) >= 2 and len(window_price) >= 2:
            corr = window_volume.corr(window_price)
            volume_momentum_corr.iloc[i] = corr if not pd.isna(corr) else 0
    
    # Strength of alignment signal
    alignment_strength = abs(volume_momentum_corr)
    
    # 4. Generate Composite Alpha
    # Base signal: Range efficiency * Momentum
    base_signal = intraday_efficiency * momentum_convergence
    
    # Adjust by volume alignment
    volume_adjusted = base_signal * alignment_strength
    
    # Apply regime-based scaling using volatility
    # Calculate 20-day rolling volatility
    returns = data['close'].pct_change()
    volatility = returns.rolling(window=20, min_periods=10).std()
    
    # Different scaling for high/low volatility regimes
    vol_threshold = volatility.quantile(0.7)  # 70th percentile as high vol threshold
    high_vol_regime = volatility > vol_threshold
    
    # Scale down in high volatility, scale up in low volatility
    regime_scaling = np.where(high_vol_regime, 0.7, 1.3)
    
    # Final signal with regime adjustment
    final_signal = volume_adjusted * regime_scaling
    
    # Apply bounded output range (-2 to 2)
    signal_std = final_signal.rolling(window=50, min_periods=20).std()
    bounded_signal = final_signal / (signal_std + 1e-8)
    bounded_signal = np.clip(bounded_signal, -2, 2)
    
    return bounded_signal
