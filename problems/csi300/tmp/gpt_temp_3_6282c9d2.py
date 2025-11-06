import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining momentum acceleration, range efficiency, 
    volume-confirmed reversal, amount flow persistence, and regime-adaptive volume clustering.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Initialize result series
    alpha = pd.Series(index=data.index, dtype=float)
    
    # 1. Momentum Acceleration with Volume Divergence
    # Multi-timeframe momentum
    mom_5d = data['close'] / data['close'].shift(5) - 1
    mom_10d = data['close'] / data['close'].shift(10) - 1
    mom_acceleration = (mom_5d / mom_10d) - 1
    
    # Volume divergence
    vol_5d_ratio = data['volume'] / data['volume'].shift(5)
    vol_10d_ratio = data['volume'] / data['volume'].shift(10)
    vol_acceleration = (vol_5d_ratio / vol_10d_ratio) - 1
    
    # Combined divergence
    price_vol_divergence = mom_acceleration - vol_acceleration
    
    # Volatility scaling for momentum component
    vol_10d = (data['high'].rolling(10).max() - data['low'].rolling(10).min()) / data['close'].shift(10)
    momentum_signal = price_vol_divergence * vol_10d
    
    # 2. Volatility-Scaled Range Efficiency
    # Multi-period range
    tr_3d = data['high'].rolling(3).max() - data['low'].rolling(3).min()
    tr_5d = data['high'].rolling(5).max() - data['low'].rolling(5).min()
    range_expansion = tr_3d / tr_5d
    
    # Price movement efficiency
    net_move_3d = abs(data['close'] / data['close'].shift(3) - 1)
    efficiency_ratio = net_move_3d / tr_3d
    efficiency_persistence = efficiency_ratio / efficiency_ratio.shift(1)
    
    # Volatility scaling for efficiency
    volatility_10d = (data['high'].rolling(10).max() - data['low'].rolling(10).min()) / data['close'].shift(10)
    scaled_efficiency = efficiency_ratio * volatility_10d
    
    # 3. Volume-Confirmed Extreme Reversal
    # Extreme move identification
    price_extreme = (data['close'] - data['low'].rolling(3).min()) / (data['high'].rolling(3).max() - data['low'].rolling(3).min())
    vol_extreme = data['volume'] / data['volume'].rolling(6).median()
    extreme_score = price_extreme * vol_extreme
    
    # Reversal timing
    vol_reversal_conf = data['volume'] / data['volume'].shift(1)
    price_reversal_mag = abs(data['close'] / data['close'].shift(1) - 1)
    
    # Multi-timeframe confirmation
    reversal_signal = extreme_score * vol_reversal_conf * price_reversal_mag
    
    # 4. Amount Flow Direction Persistence
    # Directional flow classification
    up_amount = np.where(data['close'] > data['close'].shift(1), data['amount'], 0)
    down_amount = np.where(data['close'] < data['close'].shift(1), data['amount'], 0)
    
    # Flow momentum
    net_flow_3d = pd.Series(up_amount, index=data.index).rolling(3).sum() - pd.Series(down_amount, index=data.index).rolling(3).sum()
    
    # Flow direction consistency
    flow_direction = np.sign(net_flow_3d)
    flow_consistency = flow_direction.rolling(3).apply(lambda x: len(set(x.dropna())) == 1 if len(x.dropna()) == 3 else np.nan)
    
    # Persistence signal
    flow_signal = net_flow_3d * flow_consistency
    
    # 5. Regime-Adaptive Volume Clustering
    # Volatility regime
    short_vol = (data['high'].rolling(5).max() - data['low'].rolling(5).min()) / data['close'].shift(5)
    long_vol = (data['high'].rolling(10).max() - data['low'].rolling(10).min()) / data['close'].shift(10)
    vol_regime = short_vol / long_vol
    
    # Volume pattern analysis
    vol_spike = data['volume'] / data['volume'].rolling(10).mean()
    vol_clustering = data['volume'].rolling(5).apply(lambda x: sum(x > x.mean()) if len(x.dropna()) == 5 else np.nan)
    vol_trend = data['volume'] / data['volume'].shift(5)
    
    # Adaptive signal generation
    regime_signal = np.where(
        vol_regime > 1.2,  # High volatility regime
        vol_clustering * vol_trend,  # Trend continuation
        vol_spike * (1 - vol_trend)  # Potential reversal
    )
    
    # Combine all signals with appropriate weights
    alpha = (
        0.3 * momentum_signal + 
        0.25 * scaled_efficiency + 
        0.2 * reversal_signal + 
        0.15 * flow_signal + 
        0.1 * regime_signal
    )
    
    # Normalize the final alpha
    alpha = (alpha - alpha.rolling(20).mean()) / alpha.rolling(20).std()
    
    return alpha
