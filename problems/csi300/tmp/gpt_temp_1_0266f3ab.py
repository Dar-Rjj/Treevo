import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility-normalized price-volume divergence,
    regime-aware range efficiency, volume-scaled mean reversion, order flow persistence,
    and volatility-clustered volume signals.
    """
    # Create copy to avoid modifying original data
    data = df.copy()
    
    # Volatility-Normalized Price-Volume Divergence
    # Multi-Timeframe Price Momentum
    data['momentum_5d'] = data['close'].pct_change(5)
    data['momentum_20d'] = data['close'].pct_change(20)
    data['momentum_ratio'] = data['momentum_5d'] / (data['momentum_20d'] + 1e-8)
    
    # Volume Persistence Analysis
    data['volume_trend'] = data['volume'].rolling(window=10).apply(
        lambda x: stats.linregress(range(len(x)), x)[0], raw=False
    )
    data['volume_acceleration'] = data['volume_trend'].diff()
    data['volume_regime'] = (data['volume'] > data['volume'].rolling(20).mean()).astype(int)
    
    # Normalize momentum by volatility and weight by volume persistence
    data['volatility_20d'] = data['close'].pct_change().rolling(20).std()
    data['norm_momentum'] = data['momentum_ratio'] / (data['volatility_20d'] + 1e-8)
    data['divergence_signal'] = data['norm_momentum'] * (1 + data['volume_trend'].abs())
    
    # Regime-Aware Range Efficiency
    # Volatility-Adjusted True Range
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    data['atr_10d'] = data['true_range'].rolling(10).mean()
    data['norm_range'] = data['true_range'] / (data['atr_10d'] + 1e-8)
    
    # Price Movement Efficiency
    data['abs_return'] = abs(data['close'].pct_change())
    data['efficiency'] = data['abs_return'] / (data['true_range'] + 1e-8)
    
    # Regime Detection
    data['vol_regime'] = (data['volatility_20d'] > data['volatility_20d'].rolling(50).mean()).astype(int)
    data['regime_adjusted_efficiency'] = data['efficiency'] * (1 + 0.5 * data['vol_regime'])
    
    # Volume-Scaled Mean Reversion
    # Extreme Move Detection
    data['return_zscore'] = data['close'].pct_change().rolling(10).apply(
        lambda x: (x[-1] - x.mean()) / (x.std() + 1e-8), raw=False
    )
    data['volume_zscore'] = (data['volume'] - data['volume'].rolling(20).mean()) / (data['volume'].rolling(20).std() + 1e-8)
    data['extreme_move'] = (abs(data['return_zscore']) > 2).astype(int)
    data['oversold'] = (data['return_zscore'] < -2).astype(int)
    data['overbought'] = (data['return_zscore'] > 2).astype(int)
    
    # Volume-Weighted Reversion Signal
    data['reversion_signal'] = -data['return_zscore'] * data['volume_zscore'] * data['extreme_move']
    data['reversion_signal'] = data['reversion_signal'] / (data['volatility_20d'] + 1e-8)
    
    # Order Flow Persistence Momentum
    # Amount-Based Flow Direction
    data['direction'] = (data['close'] > data['prev_close']).astype(int)
    data['directional_amount'] = data['amount'] * (2 * data['direction'] - 1)
    data['net_flow_5d'] = data['directional_amount'].rolling(5).sum()
    
    # Flow Persistence Analysis
    data['consecutive_days'] = data['direction'].groupby(
        (data['direction'] != data['direction'].shift()).cumsum()
    ).cumcount() + 1
    data['flow_momentum'] = data['net_flow_5d'].diff()
    data['flow_regime_shift'] = (abs(data['flow_momentum']) > data['flow_momentum'].rolling(10).std() * 2).astype(int)
    
    # Volatility-Clustered Volume Signals
    # Multi-Scale Volatility Measurement
    data['vol_short'] = (data['high'] - data['low']).rolling(5).mean()
    data['vol_medium'] = data['close'].pct_change().rolling(20).std()
    data['vol_ratio'] = data['vol_short'] / (data['vol_medium'] + 1e-8)
    
    # Volume Pattern Analysis
    data['high_volume_days'] = (data['volume'] > data['volume'].rolling(20).mean() * 1.2).astype(int)
    data['volume_cluster'] = data['high_volume_days'].rolling(3).sum()
    data['volume_vol_relationship'] = data['volume'] / (data['vol_medium'] + 1e-8)
    
    # Combine all signals into final alpha factor
    alpha = (
        0.3 * data['divergence_signal'] +
        0.25 * data['regime_adjusted_efficiency'] +
        0.2 * data['reversion_signal'] +
        0.15 * data['net_flow_5d'] / (data['amount'].rolling(5).mean() + 1e-8) +
        0.1 * data['volume_vol_relationship'] * data['vol_ratio']
    )
    
    # Normalize the final alpha factor
    alpha = (alpha - alpha.rolling(50).mean()) / (alpha.rolling(50).std() + 1e-8)
    
    return alpha
