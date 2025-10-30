import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Weighted Intraday Momentum Efficiency with Regime Detection
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic components
    df['intraday_momentum'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['volume_20d_avg'] = df['volume'].rolling(window=20, min_periods=10).mean()
    df['amount_5d_avg'] = df['amount'].rolling(window=5, min_periods=3).mean()
    
    # Volume-weighted adjustment
    df['volume_weighted_momentum'] = df['intraday_momentum'] * (df['volume'] / df['volume_20d_avg'])
    
    # Efficiency score with amount adjustment
    df['efficiency_score'] = df['volume_weighted_momentum'] * (1 + (df['amount'] / df['amount_5d_avg']))
    
    # Calculate regime percentiles
    df['efficiency_percentile'] = df['efficiency_score'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 10 else np.nan, raw=False
    )
    
    # Volume and amount percentiles for regime detection
    df['volume_percentile'] = df['volume'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 10 else np.nan, raw=False
    )
    df['amount_percentile'] = df['amount'].rolling(window=20, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) >= 10 else np.nan, raw=False
    )
    
    # Multi-timeframe efficiency scores
    df['efficiency_1d'] = df['efficiency_score']
    df['efficiency_3d'] = df['efficiency_score'].rolling(window=3, min_periods=2).mean()
    df['efficiency_5d'] = df['efficiency_score'].rolling(window=5, min_periods=3).mean()
    df['efficiency_10d'] = df['efficiency_score'].rolling(window=10, min_periods=5).mean()
    
    # Regime classification
    conditions = [
        (df['efficiency_percentile'] > 0.8) & (df['volume_percentile'] > 0.8) & (df['amount_percentile'] > 0.8),
        (df['efficiency_percentile'] > 0.8) & (df['volume_percentile'] > 0.8) & (df['amount_percentile'] <= 0.8),
        (df['efficiency_percentile'] > 0.8) & (df['volume_percentile'] <= 0.8) & (df['amount_percentile'] > 0.8),
        (df['efficiency_percentile'] > 0.8) & (df['volume_percentile'] <= 0.8) & (df['amount_percentile'] <= 0.8),
        (df['efficiency_percentile'] < 0.2) & (df['volume_percentile'] > 0.8) & (df['amount_percentile'] > 0.8),
        (df['efficiency_percentile'] < 0.2) & (df['volume_percentile'] > 0.8) & (df['amount_percentile'] <= 0.8),
        (df['efficiency_percentile'] < 0.2) & (df['volume_percentile'] <= 0.8) & (df['amount_percentile'] > 0.8),
        (df['efficiency_percentile'] < 0.2) & (df['volume_percentile'] <= 0.8) & (df['amount_percentile'] <= 0.8)
    ]
    
    regime_scores = [2.0, 1.5, 1.0, 0.5, -1.0, -1.5, -0.5, -2.0]
    df['regime_score'] = np.select(conditions, regime_scores, default=0.0)
    
    # Multi-timeframe confirmation signals
    df['short_term_trend'] = np.sign(df['efficiency_1d'] - df['efficiency_3d'])
    df['medium_term_trend'] = np.sign(df['efficiency_5d'] - df['efficiency_10d'])
    
    # Final factor calculation
    df['factor'] = (
        df['efficiency_score'] * 
        df['regime_score'] * 
        (1 + 0.2 * df['short_term_trend']) * 
        (1 + 0.1 * df['medium_term_trend'])
    )
    
    # Clean up intermediate columns
    result = df['factor'].copy()
    result = result.replace([np.inf, -np.inf], np.nan)
    
    return result
