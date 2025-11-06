import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Intraday Strength Ratio
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # 2. Calculate Short-Term Persistence
    data['short_term_strength'] = data['intraday_strength'].rolling(window=3, min_periods=1).mean()
    
    # Calculate consecutive days with same directional sign
    data['strength_sign'] = np.sign(data['intraday_strength'])
    data['sign_change'] = data['strength_sign'] != data['strength_sign'].shift(1)
    data['consecutive_days'] = data.groupby((data['sign_change']).cumsum()).cumcount() + 1
    
    # 3. Calculate Medium-Term Trend
    data['medium_term_strength'] = data['intraday_strength'].rolling(window=8, min_periods=1).mean()
    
    # 4. Compute Intraday Momentum Divergence
    data['momentum_divergence'] = data['short_term_strength'] - data['medium_term_strength']
    
    # 5. Assess Divergence Significance
    # Calculate historical range of divergence
    data['divergence_range'] = data['momentum_divergence'].rolling(window=20, min_periods=1).apply(
        lambda x: np.percentile(x, 80) - np.percentile(x, 20), raw=True
    )
    data['divergence_significance'] = np.abs(data['momentum_divergence']) / (data['divergence_range'] + 1e-8)
    
    # 6. Calculate True Range-Based Volatility
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    data['volatility_20d'] = data['true_range'].rolling(window=20, min_periods=1).mean()
    
    # 7. Classify Volatility Regime
    data['volatility_percentile'] = data['volatility_20d'].rolling(window=60, min_periods=1).apply(
        lambda x: (x.iloc[-1] > np.percentile(x, 80)) * 2 + (x.iloc[-1] > np.percentile(x, 20)) * 1,
        raw=False
    )
    
    # 8. Generate Volatility-Adjusted Signal
    conditions = [
        data['volatility_percentile'] == 3,  # High volatility (>80th percentile)
        data['volatility_percentile'] == 1,  # Low volatility (<20th percentile)
    ]
    choices = [
        data['momentum_divergence'] * 0.5,   # Reduce by 50% in high volatility
        data['momentum_divergence'] * 1.5,   # Amplify by 50% in low volatility
    ]
    data['volatility_adjusted_signal'] = np.select(conditions, choices, default=data['momentum_divergence'])
    
    # Apply significance filter - only keep meaningful divergences
    significant_mask = data['divergence_significance'] > 0.5
    data['final_signal'] = data['volatility_adjusted_signal'] * significant_mask
    
    # Clean up intermediate columns
    result = data['final_signal']
    result.name = 'volatility_adjusted_intraday_momentum_divergence'
    
    return result
