import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum-Efficiency Divergence with Volume Breakout alpha factor
    """
    data = df.copy()
    
    # Intraday Momentum Components
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['momentum_5d_ema'] = data['intraday_momentum'].ewm(span=5, adjust=False).mean()
    data['momentum_persistence'] = data['momentum_5d_ema'].diff().rolling(window=3).apply(lambda x: np.sum(x > 0) - np.sum(x < 0))
    
    # Price Efficiency Metrics
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    data['efficiency_ratio'] = abs(data['close'] - data['open']) / data['true_range'].replace(0, np.nan)
    data['efficiency_5d_change'] = data['efficiency_ratio'].diff(5)
    data['efficiency_trend'] = data['efficiency_ratio'].rolling(window=3).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
    
    # Intraday Divergence Patterns
    data['bullish_divergence'] = ((data['momentum_5d_ema'] < 0) & (data['efficiency_trend'] > 0)).astype(int)
    data['bearish_divergence'] = ((data['momentum_5d_ema'] > 0) & (data['efficiency_trend'] < 0)).astype(int)
    data['divergence_magnitude'] = abs(data['momentum_5d_ema'] * data['efficiency_5d_change'])
    
    # Volume Acceleration Profile
    data['volume_pct_change'] = data['volume'].pct_change()
    data['volume_acceleration'] = data['volume_pct_change'].diff(3)
    data['volume_momentum_dir'] = np.sign(data['volume_pct_change'].rolling(window=3).mean())
    data['volume_momentum_strength'] = abs(data['volume_pct_change'].rolling(window=3).mean())
    
    # Volume Breakout Strength Analysis
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_breakout_ratio'] = data['volume'] / data['volume_20d_avg']
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Volume persistence (consecutive same-sign volume days)
    def volume_persistence(series):
        signs = np.sign(series)
        current_sign = signs.iloc[-1]
        count = 0
        for i in range(len(signs)-1, -1, -1):
            if signs.iloc[i] == current_sign and current_sign != 0:
                count += 1
            else:
                break
        return count * current_sign
    
    data['volume_persistence'] = data['volume_pct_change'].rolling(window=10).apply(volume_persistence, raw=False)
    
    # Volume-Phase Shift Detection
    data['momentum_decay_rate'] = data['momentum_5d_ema'].diff().rolling(window=5).mean()
    data['volume_phase_shift'] = (data['volume_acceleration'] * data['momentum_decay_rate']).rolling(window=5).mean()
    data['volume_momentum_consistency'] = (data['volume_momentum_dir'] * np.sign(data['momentum_5d_ema'])).rolling(window=5).mean()
    
    # Range Compression Analysis
    data['daily_range'] = data['high'] - data['low']
    data['range_20d_avg'] = data['daily_range'].rolling(window=20).mean()
    data['range_compression_ratio'] = data['daily_range'] / data['range_20d_avg']
    
    # Consecutive compression days tracking
    def compression_streak(series):
        count = 0
        for i in range(len(series)-1, -1, -1):
            if series.iloc[i] < 1.0:
                count += 1
            else:
                break
        return count
    
    data['compression_duration'] = data['range_compression_ratio'].rolling(window=20).apply(compression_streak, raw=False)
    data['compression_severity'] = (1 - data['range_compression_ratio']).rolling(window=5).mean()
    
    # Volume Behavior During Compression
    data['compression_volume_ratio'] = data['volume'] / data['volume'].rolling(window=20).mean()
    data['volume_compression_trend'] = data['volume_pct_change'].rolling(window=5).mean()
    
    # Breakout Probability Assessment
    data['breakout_probability'] = (data['compression_duration'] * data['volume_breakout_ratio']).rolling(window=10).mean()
    data['projected_breakout_mag'] = data['divergence_magnitude'] * data['volume_breakout_ratio']
    
    # Composite Alpha Signal Generation
    # Core Divergence Component
    data['divergence_score'] = (
        data['bullish_divergence'] - data['bearish_divergence']
    ) * data['divergence_magnitude'] * (1 + data['volume_momentum_consistency'])
    
    # Volume Breakout Enhancement
    data['volume_enhancement'] = (
        data['volume_acceleration'] * data['volume_breakout_ratio'] * 
        np.sign(data['volume_persistence'])
    )
    
    # Range Compression Integration
    data['compression_factor'] = (
        data['compression_duration'] * data['compression_severity'] * 
        data['breakout_probability'] * (1 + data['volume_compression_trend'])
    )
    
    # Final Alpha Factor
    alpha_signal = (
        data['divergence_score'] * 
        (1 + data['volume_enhancement']) * 
        (1 + data['compression_factor'])
    )
    
    # Enhanced signals during range compression periods
    compression_mask = data['range_compression_ratio'] < 1.0
    alpha_signal = np.where(
        compression_mask,
        alpha_signal * (1 + data['compression_severity']),
        alpha_signal
    )
    
    return alpha_signal
