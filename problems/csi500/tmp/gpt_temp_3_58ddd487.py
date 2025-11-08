import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Microstructure Momentum factor
    """
    # Compute Microstructure Momentum
    # Volume-weighted price velocity
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    df['price_velocity'] = df['close'].diff() / df['close'].shift(1)
    df['volume_weighted_velocity'] = df['price_velocity'] * (df['volume'] / df['volume'].rolling(window=20).mean())
    
    # Microstructure noise ratio
    df['intraday_range'] = (df['high'] - df['low']) / df['close']
    df['noise_ratio'] = df['intraday_range'].rolling(window=10).std() / df['intraday_range'].rolling(window=50).std()
    
    # Pure momentum component
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    df['pure_momentum'] = (df['momentum_5'] + df['momentum_10']) / 2 * (1 - df['noise_ratio'])
    
    # Measure Volatility Regime Adaptation
    # Volatility bands
    df['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
    df['volatility_50'] = df['close'].pct_change().rolling(window=50).std()
    df['vol_regime'] = df['volatility_20'] / df['volatility_50']
    
    # Volatility persistence
    df['vol_persistence'] = df['vol_regime'].rolling(window=10).corr(df['vol_regime'].shift(5))
    df['vol_persistence'] = df['vol_persistence'].fillna(0)
    
    # Regime transition probability
    df['regime_change'] = (df['vol_regime'].diff() > df['vol_regime'].rolling(window=20).std()).astype(int)
    df['transition_prob'] = df['regime_change'].rolling(window=10).mean()
    
    # Incorporate Order Flow Asymmetry
    # Directional order flow intensity
    df['dollar_volume'] = df['close'] * df['volume']
    df['flow_intensity'] = df['dollar_volume'].diff() / df['dollar_volume'].rolling(window=20).mean()
    
    # Flow persistence gradient
    df['flow_persistence'] = df['flow_intensity'].rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
    )
    
    # Flow reversal patterns
    df['flow_reversal'] = (df['flow_intensity'] * df['flow_intensity'].shift(1) < 0).astype(int)
    df['reversal_strength'] = df['flow_reversal'].rolling(window=5).mean() * abs(df['flow_intensity'])
    
    # Synthesize Adaptive Signal
    # Combine momentum with volatility regime
    df['momentum_vol_adjusted'] = df['pure_momentum'] * (1 + df['vol_regime'])
    
    # Scale by order flow asymmetry
    df['flow_asymmetry'] = df['flow_persistence'] * (1 - df['reversal_strength'])
    df['momentum_flow_scaled'] = df['momentum_vol_adjusted'] * (1 + df['flow_asymmetry'])
    
    # Apply regime-dependent transformation
    df['regime_weight'] = np.where(df['vol_regime'] > 1, 1.2, 0.8)  # High vol gets higher weight
    df['final_signal'] = df['momentum_flow_scaled'] * df['regime_weight']
    
    # Add momentum persistence filter
    df['momentum_persistence'] = df['pure_momentum'].rolling(window=5).apply(
        lambda x: np.corrcoef(range(len(x)), x)[0,1] if len(x) > 1 and not np.isnan(x).any() else 0
    )
    df['filtered_signal'] = df['final_signal'] * (1 + df['momentum_persistence'])
    
    # Clean up and return
    result = df['filtered_signal'].fillna(0)
    return result
