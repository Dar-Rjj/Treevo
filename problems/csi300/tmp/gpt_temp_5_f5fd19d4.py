import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Surface Anomaly Detection factor that identifies abnormal patterns 
    in volatility dynamics using available price and volume data
    """
    # Calculate daily returns and volatility proxies
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['high_low_range'] = (df['high'] - df['low']) / df['close']
    df['open_close_range'] = abs(df['close'] - df['open']) / df['close']
    
    # Short-term volatility (5-day rolling)
    df['short_term_vol'] = df['returns'].rolling(window=5).std()
    
    # Medium-term volatility (20-day rolling)
    df['medium_term_vol'] = df['returns'].rolling(window=20).std()
    
    # Volatility term structure divergence
    df['vol_term_divergence'] = (df['short_term_vol'] - df['medium_term_vol']) / df['medium_term_vol']
    
    # Volatility persistence measure (autocorrelation of volatility)
    df['vol_persistence'] = df['short_term_vol'].rolling(window=10).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
    )
    
    # Skew dynamics proxy using high-low vs open-close ranges
    df['range_skew'] = df['high_low_range'] - df['open_close_range']
    df['skew_persistence'] = df['range_skew'].rolling(window=10).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else np.nan, raw=False
    )
    
    # Volatility risk premium proxy (realized vs expected volatility)
    df['vol_risk_premium'] = df['short_term_vol'] - df['medium_term_vol'].shift(1)
    
    # Volume-volatility relationship anomaly
    df['volume_vol_correlation'] = df['volume'].rolling(window=10).corr(df['high_low_range'])
    
    # Volatility surface convexity proxy
    df['vol_convexity'] = (
        df['short_term_vol'].rolling(window=5).std() / 
        df['medium_term_vol'].rolling(window=5).std()
    )
    
    # Jump risk premium identification
    df['jump_risk_indicator'] = (
        (df['high_low_range'] - df['returns'].abs()) / 
        df['high_low_range'].rolling(window=10).std()
    )
    
    # Combined anomaly score
    factors = [
        'vol_term_divergence',
        'vol_persistence', 
        'skew_persistence',
        'vol_risk_premium',
        'volume_vol_correlation',
        'vol_convexity',
        'jump_risk_indicator'
    ]
    
    # Standardize each factor and combine
    anomaly_scores = []
    for factor in factors:
        if factor in df.columns:
            z_score = (df[factor] - df[factor].rolling(window=50).mean()) / df[factor].rolling(window=50).std()
            anomaly_scores.append(z_score)
    
    if anomaly_scores:
        combined_anomaly = pd.concat(anomaly_scores, axis=1).mean(axis=1)
    else:
        combined_anomaly = pd.Series(index=df.index, data=0)
    
    return combined_anomaly
