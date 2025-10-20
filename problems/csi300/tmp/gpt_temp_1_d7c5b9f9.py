import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Squeeze with Volume-Amount Confirmation
    """
    data = df.copy()
    
    # Multi-Period Momentum Analysis with Exponential Decay
    # Compute returns across multiple horizons
    data['ret_5'] = data['close'] / data['close'].shift(5) - 1
    data['ret_20'] = data['close'] / data['close'].shift(20) - 1
    data['ret_60'] = data['close'] / data['close'].shift(60) - 1
    
    # Apply exponential decay weighting (λ=0.94)
    lambda_val = 0.94
    decay_weights = np.array([lambda_val**5, lambda_val**20, lambda_val**60])
    returns_matrix = data[['ret_5', 'ret_20', 'ret_60']].fillna(0).values
    data['decay_momentum'] = np.sum(returns_matrix * decay_weights, axis=1)
    
    # Volatility Regime Detection using Multi-Indicator Approach
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate volatility metrics
    data['ret_daily'] = data['close'].pct_change()
    data['std_10'] = data['ret_daily'].rolling(window=10, min_periods=5).std()
    data['std_20'] = data['ret_daily'].rolling(window=20, min_periods=10).std()
    data['atr_5'] = data['true_range'].rolling(window=5, min_periods=3).mean()
    data['atr_20'] = data['true_range'].rolling(window=20, min_periods=10).mean()
    
    # Determine volatility regime classification
    data['regime_score'] = (data['std_10'] / data['std_20']) * (data['atr_5'] / data['atr_20'])
    data['vol_regime'] = np.select(
        [data['regime_score'] > 1.2, data['regime_score'] < 0.8],
        [2, 0],  # 2: high volatility, 0: low volatility, 1: normal
        default=1
    )
    
    # Momentum Divergence Detection with Squeeze Integration
    # Calculate decay-weighted momentum for different periods
    data['decay_momentum_5'] = data['ret_5'] * lambda_val**5
    data['decay_momentum_20'] = data['ret_20'] * lambda_val**20
    data['decay_momentum_60'] = data['ret_60'] * lambda_val**60
    
    # Primary and secondary divergence
    data['divergence_primary'] = data['decay_momentum_5'] - data['decay_momentum_20']
    data['divergence_secondary'] = data['decay_momentum_20'] - data['decay_momentum_60']
    
    # Squeeze condition analysis
    data['ma_20'] = data['close'].rolling(window=20, min_periods=10).mean()
    data['std_20_close'] = data['close'].rolling(window=20, min_periods=10).std()
    data['bb_upper'] = data['ma_20'] + 2 * data['std_20_close']
    data['bb_lower'] = data['ma_20'] - 2 * data['std_20_close']
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['ma_20']
    data['bb_width_percentile'] = data['bb_width'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['squeeze'] = (data['bb_width_percentile'] < 0.2).astype(int)
    
    # Volume-Amount Confirmation System
    # Volume momentum analysis
    data['volume_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_20'] = data['volume'].rolling(window=20, min_periods=10).mean()
    data['volume_60'] = data['volume'].rolling(window=60, min_periods=30).mean()
    
    data['volume_ratio_5_20'] = data['volume_5'] / data['volume_20']
    data['volume_ratio_20_60'] = data['volume_20'] / data['volume_60']
    data['volume_acceleration'] = data['volume_ratio_5_20'].diff(5)
    
    # Amount validation
    data['amount_5'] = data['amount'].rolling(window=5, min_periods=3).mean()
    data['amount_volume_ratio'] = data['amount_5'] / data['volume_5']
    data['avr_median_20'] = data['amount_volume_ratio'].rolling(window=20, min_periods=10).median()
    data['large_order_flow'] = (data['amount_volume_ratio'] > data['avr_median_20']).astype(int)
    
    # Composite Alpha Synthesis with Regime Optimization
    # Base factor construction
    data['base_factor'] = data['decay_momentum'] * (1 + data['divergence_primary'] + 0.5 * data['divergence_secondary'])
    
    # Apply volatility regime scaling
    regime_scaling = np.select(
        [data['vol_regime'] == 0, data['vol_regime'] == 2],
        [1.5, 0.8],  # 1.5x in low volatility, 0.8x in high volatility
        default=1.0
    )
    data['regime_scaled_factor'] = data['base_factor'] * regime_scaling
    
    # Squeeze amplification
    squeeze_amplification = np.where(
        data['squeeze'] == 1,
        1 / (data['bb_width'] + 1e-6),  # Inverse Bollinger Band width during squeeze
        1.0
    )
    data['squeeze_amplified'] = data['regime_scaled_factor'] * squeeze_amplification
    
    # Volume-amount confirmation
    volume_confirmation = (1 + data['volume_ratio_5_20'] * data['volume_acceleration'].clip(lower=0))
    amount_confirmation = (1 + 0.5 * data['large_order_flow'])
    
    # Apply confirmation strength based on regime classification
    regime_confirmation = np.select(
        [data['vol_regime'] == 0, data['vol_regime'] == 2],
        [1.2, 0.9],  # Stronger confirmation in low volatility
        default=1.0
    )
    
    # Final alpha factor
    data['alpha_factor'] = (
        data['squeeze_amplified'] * 
        volume_confirmation * 
        amount_confirmation * 
        regime_confirmation
    )
    
    return data['alpha_factor']
