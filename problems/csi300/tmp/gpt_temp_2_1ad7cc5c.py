import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price Gap Reversion with Liquidity Acceleration alpha factor
    """
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['overnight_gap'] = data['open'] / data['prev_close'] - 1
    
    # Calculate True Range and ATR
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_5'] = data['true_range'].rolling(window=5, min_periods=3).mean()
    
    # Gap magnitude relative to ATR
    data['gap_atr_ratio'] = abs(data['overnight_gap']) / data['atr_5']
    
    # Gap percentile in 20-day window
    data['gap_percentile'] = data['overnight_gap'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False
    )
    
    # Gap persistence analysis
    data['gap_fill_speed'] = np.where(
        data['overnight_gap'] > 0,
        (data['high'] - data['prev_close']) / (data['open'] - data['prev_close']),
        (data['prev_close'] - data['low']) / (data['prev_close'] - data['open'])
    )
    
    # Intraday reversion from open
    data['high_low_ratio'] = np.where(
        data['open'] != data['low'],
        (data['high'] - data['open']) / (data['open'] - data['low']),
        1.0
    )
    data['close_open_return'] = data['close'] / data['open'] - 1
    
    # Gap reversion strength
    data['gap_reversion_strength'] = np.where(
        data['overnight_gap'] * data['close_open_return'] < 0,
        abs(data['close_open_return']) / (abs(data['overnight_gap']) + 1e-8),
        0
    )
    
    # Volume acceleration patterns
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_burst_ratio'] = data['volume'] / data['volume_ma_5']
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(1) - 1
    
    # Amount-based liquidity signals
    data['amount_velocity'] = data['amount'] / data['amount'].shift(1) - 1
    data['amount_acceleration'] = data['amount_velocity'] - data['amount_velocity'].shift(1)
    
    # Amount efficiency
    data['price_change'] = abs(data['close'] - data['prev_close'])
    data['amount_per_price_change'] = np.where(
        data['price_change'] > 0,
        data['amount'] / data['price_change'],
        data['amount']
    )
    data['amount_per_volume'] = np.where(
        data['volume'] > 0,
        data['amount'] / data['volume'],
        0
    )
    
    # Liquidity quality score
    data['amount_ma_10'] = data['amount'].rolling(window=10, min_periods=5).mean()
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['liquidity_quality'] = (
        (data['amount'] / data['amount_ma_10']) * 
        (data['volume'] / data['volume_ma_10']) *
        (1 + data['amount_per_volume'] / data['amount_per_volume'].rolling(window=20, min_periods=10).mean())
    )
    
    # Gap reversion consistency
    reversion_signal = (data['overnight_gap'] * data['close_open_return'] < 0).astype(int)
    data['consecutive_reversions'] = reversion_signal * (reversion_signal.groupby((reversion_signal != reversion_signal.shift()).cumsum()).cumcount() + 1)
    
    # Combine gap and liquidity signals
    gap_signal = (
        data['gap_reversion_strength'] * 
        (1 + data['consecutive_reversions'] / 5) *
        np.where(data['gap_fill_speed'] < 2, 1, 0.5)  # Penalize too fast gap filling
    )
    
    liquidity_signal = (
        data['volume_burst_ratio'] * 
        data['liquidity_quality'] *
        (1 + np.tanh(data['amount_acceleration']))
    )
    
    # Generate alpha factor
    alpha = gap_signal * liquidity_signal
    
    # Apply dynamic threshold adjustment based on recent volatility
    volatility_20 = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    volatility_adjustment = 1 / (1 + volatility_20)
    alpha = alpha * volatility_adjustment
    
    # Signal smoothing and risk control
    alpha_smoothed = alpha.rolling(window=3, min_periods=2).mean()
    
    # Factor volatility control
    alpha_std = alpha_smoothed.rolling(window=20, min_periods=10).std()
    alpha_normalized = np.where(
        alpha_std > 0,
        alpha_smoothed / alpha_std,
        alpha_smoothed
    )
    
    # Final alpha factor with position sizing constraint
    final_alpha = np.tanh(alpha_normalized / 3)  # Constrain to [-1, 1]
    
    return final_alpha
