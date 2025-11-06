import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Hierarchical Price Reversal Patterns with Liquidity Dynamics
    Combines extreme price movement detection with liquidity analysis
    to identify high-probability reversal opportunities
    """
    data = df.copy()
    
    # Multi-Dimensional Reversal Signal Detection
    # Extreme Price Movement Identification
    data['intraday_range'] = data['high'] - data['low']
    data['range_avg_4d'] = data['intraday_range'].rolling(window=4, min_periods=3).mean()
    data['range_expansion'] = data['intraday_range'] / data['range_avg_4d'].shift(1)
    
    # Close-to-Close Momentum Extremes
    data['close_ret'] = data['close'] / data['close'].shift(1) - 1
    data['ret_percentile'] = data['close_ret'].rolling(window=20, min_periods=15).apply(
        lambda x: (x[-1] - x[:-1].mean()) / x[:-1].std() if len(x) > 1 and x[:-1].std() > 0 else 0
    )
    
    # Opening Gap Reversal
    data['gap'] = data['open'] / data['close'].shift(1) - 1
    data['gap_volatility'] = data['gap'].abs().rolling(window=10, min_periods=8).std()
    data['normalized_gap'] = data['gap'] / (data['gap_volatility'].shift(1) + 1e-8)
    
    # Reversal Confirmation Framework
    # Price Action Patterns
    data['body'] = data['close'] - data['open']
    data['upper_shadow'] = data['high'] - data[['open', 'close']].max(axis=1)
    data['lower_shadow'] = data[['open', 'close']].min(axis=1) - data['low']
    
    # Hammer pattern detection
    data['hammer'] = (
        (data['lower_shadow'] > 2 * data['body'].abs()) & 
        (data['upper_shadow'] < 0.3 * data['body'].abs()) &
        (data['body'] > 0)
    ).astype(int)
    
    # Shooting star pattern detection
    data['shooting_star'] = (
        (data['upper_shadow'] > 2 * data['body'].abs()) & 
        (data['lower_shadow'] < 0.3 * data['body'].abs()) &
        (data['body'] < 0)
    ).astype(int)
    
    # Engulfing pattern
    data['prev_body'] = (data['close'].shift(1) - data['open'].shift(1)).abs()
    data['engulfing'] = (
        (data['body'].abs() > data['prev_body']) &
        ((data['body'] * (data['close'].shift(1) - data['open'].shift(1))) < 0)
    ).astype(int)
    
    # Support/Resistance Break Failure
    data['resistance_level'] = data['high'].rolling(window=10, min_periods=8).max()
    data['support_level'] = data['low'].rolling(window=10, min_periods=8).min()
    data['break_failure'] = (
        (data['high'] > data['resistance_level'].shift(1)) & 
        (data['close'] < data['resistance_level'].shift(1))
    ).astype(int) - (
        (data['low'] < data['support_level'].shift(1)) & 
        (data['close'] > data['support_level'].shift(1))
    ).astype(int)
    
    # Liquidity Flow Analysis
    # Transaction-Based Liquidity
    data['amount_per_trade'] = data['amount'] / (data['volume'] + 1e-8)
    data['amt_per_trade_avg'] = data['amount_per_trade'].rolling(window=10, min_periods=8).mean()
    data['liquidity_stress'] = data['amount_per_trade'] / (data['amt_per_trade_avg'].shift(1) + 1e-8)
    
    # Large Trade Concentration (proxy using volume distribution)
    data['volume_zscore'] = (
        data['volume'] - data['volume'].rolling(window=20, min_periods=15).mean()
    ) / (data['volume'].rolling(window=20, min_periods=15).std() + 1e-8)
    data['large_trade_concentration'] = data['volume_zscore'].clip(lower=0)
    
    # Market Participation Quality
    data['volume_persistence'] = (
        data['volume'] > data['volume'].rolling(window=5, min_periods=4).mean()
    ).rolling(window=3, min_periods=2).sum()
    
    # Adaptive Signal Generation
    # Reversal Probability Score
    data['extreme_price_signal'] = (
        (data['range_expansion'] > 1.5).astype(int) +
        (data['ret_percentile'].abs() > 1.5).astype(int) +
        (data['normalized_gap'].abs() > 2).astype(int)
    )
    
    data['pattern_confirmation'] = (
        data['hammer'] + data['shooting_star'] + data['engulfing'] + 
        data['break_failure'].abs()
    )
    
    data['reversal_probability'] = (
        data['extreme_price_signal'] * 0.4 + 
        data['pattern_confirmation'] * 0.6
    )
    
    # Weight by liquidity stress indicators
    data['liquidity_weight'] = (
        (data['liquidity_stress'] < 0.8).astype(int) * 0.7 +  # Poor liquidity
        (data['large_trade_concentration'] > 1).astype(int) * 0.3  # Concentrated trades
    )
    
    # Final Alpha Factor Construction
    data['alpha_factor'] = (
        # Strong reversal + poor liquidity → High probability reversal
        (data['reversal_probability'] >= 2) & (data['liquidity_stress'] < 0.8)
    ).astype(int) * 1.0 + \
    (
        # Weak reversal + strong liquidity → Continuation likely (negative signal)
        (data['reversal_probability'] <= 1) & (data['liquidity_stress'] >= 1.2)
    ).astype(int) * -0.5 + \
    (
        # Mixed reversal + concentrated trades → Institutional positioning
        (data['reversal_probability'] == 1) & (data['large_trade_concentration'] > 1)
    ).astype(int) * 0.3
    
    # Normalize and smooth the final factor
    alpha_series = data['alpha_factor'].fillna(0)
    alpha_series = alpha_series - alpha_series.rolling(window=10, min_periods=8).mean()
    alpha_series = alpha_series / (alpha_series.rolling(window=20, min_periods=15).std() + 1e-8)
    
    return alpha_series
