import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining volatility-scaled momentum divergence,
    volume acceleration, regime-adaptive reversal signals, and multi-timeframe divergence.
    """
    # Copy the dataframe to avoid modifying the original
    data = df.copy()
    
    # Volatility-Scaled Momentum Divergence
    # Multi-Timeframe Momentum Calculation
    data['st_momentum'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['mt_momentum'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    data['momentum_divergence'] = data['st_momentum'] - data['mt_momentum']
    
    # Dynamic Volatility Scaling
    # True Range Calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['volatility_10d'] = data['true_range'].rolling(window=10).mean()
    
    # Volatility-scaled momentum divergence
    data['vol_scaled_momentum_div'] = data['momentum_divergence'] / data['volatility_10d']
    
    # Volume Acceleration Factor
    # Volume Momentum Calculation
    data['vol_accel_1d'] = data['volume'] / data['volume'].shift(1) - 1
    data['vol_accel_2d'] = data['volume'].shift(1) / data['volume'].shift(2) - 1
    data['vol_trend_10d'] = data['volume'] / data['volume'].shift(10)
    
    # Short-term volume acceleration (average of 1-day and 2-day)
    data['short_term_vol_accel'] = (data['vol_accel_1d'] + data['vol_accel_2d']) / 2
    
    # Price-Volume Relationship
    # 5-day rolling correlation between volume changes and price returns
    data['price_return'] = data['close'].pct_change()
    data['vol_change'] = data['volume'].pct_change()
    
    # Calculate rolling correlation
    vol_price_corr = []
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            corr = window_data['vol_change'].corr(window_data['price_return'])
            vol_price_corr.append(corr if not np.isnan(corr) else 0)
        else:
            vol_price_corr.append(0)
    data['vol_price_corr_5d'] = vol_price_corr
    
    # Volume Acceleration Score
    data['volume_accel_score'] = data['short_term_vol_accel'] * data['vol_price_corr_5d']
    
    # Regime-Adaptive Reversal Factor
    # Market Regime Detection
    data['vol_regime'] = np.where(
        data['true_range'].rolling(window=10).mean() > data['true_range'].rolling(window=20).mean(),
        'high_vol', 'low_vol'
    )
    
    # Trend Regime Identification
    data['trend_regime'] = np.where(
        data['mt_momentum'] > data['st_momentum'], 'uptrend',
        np.where(data['mt_momentum'] < data['st_momentum'], 'downtrend', 'sideways')
    )
    
    # Price extremity (distance from recent high/low)
    data['recent_high'] = data['high'].rolling(window=5).max()
    data['recent_low'] = data['low'].rolling(window=5).min()
    data['price_extremity'] = np.where(
        abs(data['close'] - data['recent_high']) < abs(data['close'] - data['recent_low']),
        (data['close'] - data['recent_high']) / data['recent_high'],
        (data['close'] - data['recent_low']) / data['recent_low']
    )
    
    # Regime weights
    data['regime_weight'] = np.where(
        data['vol_regime'] == 'high_vol', 1.2,
        np.where(data['trend_regime'] == 'sideways', 1.0, 0.8)
    )
    
    # Volume confirmation (absolute volume acceleration)
    data['volume_confirmation'] = abs(data['short_term_vol_accel'])
    
    # Reversal Score
    data['reversal_score'] = data['price_extremity'] * data['regime_weight'] * data['volume_confirmation']
    
    # Multi-Timeframe Divergence Blend
    # Very short-term (1-3 days) - intraday patterns
    data['vs_momentum'] = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    data['vs_divergence'] = data['vs_momentum'] - data['st_momentum']
    
    # Short-term (5-10 days) - weekly momentum
    data['s_momentum'] = (data['close'] - data['close'].shift(8)) / data['close'].shift(8)
    data['s_divergence'] = data['s_momentum'] - data['mt_momentum']
    
    # Medium-term (15-25 days) - monthly trends
    data['m_momentum'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    data['m_divergence'] = data['m_momentum'] - data['m_momentum'].shift(5)  # vs 5 days ago
    
    # Multi-timeframe divergence with fixed weights
    data['multi_tf_divergence'] = (
        0.4 * data['vs_divergence'] + 
        0.35 * data['s_divergence'] + 
        0.25 * data['m_divergence']
    )
    
    # Final combined factor - simple average of all components
    final_factor = (
        data['vol_scaled_momentum_div'] + 
        data['volume_accel_score'] + 
        data['reversal_score'] + 
        data['multi_tf_divergence']
    ) / 4
    
    # Return the final factor series
    return final_factor
