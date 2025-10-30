import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Elasticity with Volume-Regime Adaptive Reversal factor
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Elasticity
    # Momentum Alignment Assessment
    data['momentum_3d'] = data['close'].pct_change(3)
    data['momentum_10d'] = data['close'].pct_change(10)
    
    # Momentum consistency scoring
    data['momentum_alignment'] = np.where(
        (data['momentum_3d'] * data['momentum_10d']) > 0,
        np.abs(data['momentum_3d'] + data['momentum_10d']) / 2,
        -np.abs(data['momentum_3d'] - data['momentum_10d']) / 2
    )
    
    # Volume-Price Elasticity Measurement
    data['volume_20d_avg'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_shock'] = data['volume'] / data['volume_20d_avg'] - 1
    
    # Price level elasticity calculation
    data['price_level'] = (data['close'] - data['close'].rolling(window=20).min()) / \
                         (data['close'].rolling(window=20).max() - data['close'].rolling(window=20).min())
    
    data['elasticity_coeff'] = np.where(
        data['volume_shock'] > 0,
        data['momentum_3d'] / (data['volume_shock'] + 1e-6),
        data['momentum_3d'] * (1 + np.abs(data['volume_shock']))
    )
    
    # Elasticity-Adjusted Momentum
    data['elasticity_adjusted_momentum'] = data['momentum_alignment'] * \
                                         (1 + data['elasticity_coeff'] * data['price_level'])
    
    # Intraday Efficiency & Range Analysis
    # True range calculation
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    
    # Intraday strength ratio
    data['intraday_strength'] = np.where(
        data['high'] != data['low'],
        (data['close'] - data['open']) / (data['high'] - data['low']),
        0
    )
    
    # 5-day intraday efficiency persistence
    data['efficiency_persistence'] = data['intraday_strength'].rolling(window=5).std()
    
    # Volatility-Regime Classification
    data['atr_10d'] = data['true_range'].rolling(window=10, min_periods=5).mean()
    data['volatility_regime'] = data['atr_10d'] / data['atr_10d'].rolling(window=20).mean()
    
    # Efficiency-Weighted Signals
    data['efficiency_weighted_signal'] = data['intraday_strength'] * \
                                       (1 - data['efficiency_persistence']) * \
                                       (2 - data['volatility_regime'])
    
    # Volume Acceleration Divergence
    # Volume Trend Analysis
    def volume_slope(series):
        if len(series) < 3:
            return 0
        x = np.arange(len(series))
        return np.polyfit(x, series, 1)[0]
    
    data['volume_slope'] = data['volume'].rolling(window=3).apply(volume_slope, raw=True)
    data['volume_persistence'] = (data['volume'] > data['volume'].shift(1)).rolling(window=5).sum() / 5
    
    # Momentum-Volume Divergence
    data['momentum_volume_divergence'] = np.where(
        data['volume_slope'] * data['momentum_3d'] < 0,
        -np.abs(data['momentum_3d']) * np.abs(data['volume_slope']),
        np.abs(data['momentum_3d']) * np.abs(data['volume_slope'])
    )
    
    # Volume-Regime Adaptive Logic
    data['volume_regime'] = np.select(
        [
            data['volume'] > data['volume_20d_avg'] * 1.5,
            data['volume'] < data['volume_20d_avg'] * 0.7,
        ],
        [2, 0],  # high=2, low=0
        default=1  # normal=1
    )
    
    data['volume_adaptive_signal'] = data['momentum_volume_divergence'] * data['volume_regime']
    
    # Adaptive Signal Synthesis
    # Multi-Dimensional Signal Combination
    elasticity_component = data['elasticity_adjusted_momentum'] * (2 - data['volatility_regime'])
    efficiency_component = data['efficiency_weighted_signal'] * data['volume_persistence']
    volume_component = data['volume_adaptive_signal'] * (1 + data['price_level'])
    
    # Regime-Adaptive Weighting
    volatility_weight = 1 / (1 + np.abs(data['volatility_regime'] - 1))
    volume_weight = data['volume_regime'] / 2
    price_weight = 1 - np.abs(data['price_level'] - 0.5)
    
    # Final Alpha Generation with contrarian logic
    raw_signal = (elasticity_component * volatility_weight + 
                 efficiency_component * volume_weight + 
                 volume_component * price_weight) / 3
    
    # Apply contrarian logic for extreme signals
    signal_percentile = raw_signal.rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.8)) or (x.iloc[-1] < x.quantile(0.2))
    )
    
    data['final_alpha'] = np.where(
        signal_percentile,
        -raw_signal * (1 + data['volume_persistence']),
        raw_signal * (1 + data['volume_persistence'])
    )
    
    # Volume-confidence weighted final factor
    volume_confidence = 1 - data['volume'].rolling(window=10).std() / data['volume'].rolling(window=10).mean()
    final_factor = data['final_alpha'] * np.clip(volume_confidence, 0.1, 1)
    
    return final_factor
