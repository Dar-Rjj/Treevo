import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Price-Volume Asymmetry Analysis factor
    """
    # Calculate daily returns and price movement metrics
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['daily_range'] = (df['high'] - df['low']) / df['close']
    df['abs_returns'] = abs(df['returns'])
    
    # Volatility Regime Classification
    df['volatility_20d'] = df['returns'].rolling(window=20).std()
    df['volatility_quantile'] = df['volatility_20d'].rolling(window=60).apply(
        lambda x: pd.qcut(x, 4, labels=False, duplicates='drop').iloc[-1] if len(x.dropna()) >= 40 else np.nan, 
        raw=False
    )
    
    # Volatility persistence (autocorrelation)
    df['volatility_lag1'] = df['volatility_20d'].shift(1)
    df['volatility_persistence'] = (df['volatility_20d'] - df['volatility_lag1']) / df['volatility_lag1']
    
    # Volatility breakout detection
    df['volatility_breakout'] = df['volatility_20d'] > df['volatility_20d'].rolling(window=40).quantile(0.8)
    
    # Asymmetric Volume Analysis
    df['up_day'] = df['returns'] > 0
    df['down_day'] = df['returns'] < 0
    
    # Rolling volume concentration metrics
    df['up_volume_5d'] = df.apply(lambda x: x['volume'] if x['up_day'] else 0, axis=1).rolling(window=5).sum()
    df['down_volume_5d'] = df.apply(lambda x: x['volume'] if x['down_day'] else 0, axis=1).rolling(window=5).sum()
    df['total_volume_5d'] = df['volume'].rolling(window=5).sum()
    
    df['up_volume_concentration'] = df['up_volume_5d'] / df['total_volume_5d']
    df['down_volume_intensity'] = df['down_volume_5d'] / df['total_volume_5d']
    df['volume_asymmetry_ratio'] = df['up_volume_concentration'] / (df['down_volume_intensity'] + 1e-8)
    
    # Price Movement Quality
    df['directional_efficiency'] = df['abs_returns'] / (df['daily_range'] + 1e-8)
    df['gap_behavior'] = abs(df['open'] - df['close'].shift(1)) / (df['daily_range'] + 1e-8)
    
    # Intraday momentum persistence (morning vs afternoon)
    df['morning_move'] = (df['high'].rolling(window=2).max() - df['open']) / df['open']
    df['afternoon_move'] = (df['close'] - df['open']) / df['open']
    df['intraday_momentum'] = np.sign(df['morning_move']) * np.sign(df['afternoon_move'])
    
    # Volatility-Adjusted Asymmetry Signals
    df['high_vol_asymmetry'] = ((df['volatility_quantile'] >= 2) & 
                               (df['volume_asymmetry_ratio'] > 1.2)).astype(int)
    
    df['low_vol_divergence'] = ((df['volatility_quantile'] <= 1) & 
                               ((df['volume_asymmetry_ratio'] < 0.8) | 
                                (df['volume_asymmetry_ratio'] > 1.5))).astype(int)
    
    df['regime_transition'] = ((df['volatility_persistence'].abs() > 0.3) & 
                              (df['volume_asymmetry_ratio'].pct_change().abs() > 0.4)).astype(int)
    
    # Multi-Timeframe Confirmation
    # Short-term (3-day) asymmetry patterns
    df['short_term_asymmetry'] = df['volume_asymmetry_ratio'].rolling(window=3).mean()
    df['short_term_trend'] = df['short_term_asymmetry'].pct_change(2)
    
    # Medium-term (8-day) volatility-adjusted signals
    df['medium_term_signal'] = (df['high_vol_asymmetry'].rolling(window=8).sum() - 
                               df['low_vol_divergence'].rolling(window=8).sum())
    
    # Long-term (15-day) regime consistency
    df['long_term_regime'] = df['volatility_quantile'].rolling(window=15).apply(
        lambda x: len(set(x.dropna())) == 1 if len(x.dropna()) >= 10 else np.nan, 
        raw=False
    )
    
    # Final factor construction
    factor = (
        # Base volume asymmetry signal
        df['volume_asymmetry_ratio'].rolling(window=5).mean() * 0.3 +
        
        # Volatility regime adjustment
        (df['volatility_quantile'] / 3) * df['directional_efficiency'] * 0.2 +
        
        # Multi-timeframe confirmation
        df['short_term_trend'] * 0.15 +
        df['medium_term_signal'] * 0.2 +
        df['long_term_regime'].fillna(0) * 0.15 +
        
        # Breakout and transition signals
        df['volatility_breakout'].astype(int) * df['volume_asymmetry_ratio'] * 0.1 +
        df['regime_transition'].astype(int) * df['intraday_momentum'] * 0.1
    )
    
    # Normalize the factor
    factor = (factor - factor.rolling(window=60).mean()) / (factor.rolling(window=60).std() + 1e-8)
    
    return factor
