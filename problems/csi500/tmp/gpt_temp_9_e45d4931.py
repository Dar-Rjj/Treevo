import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Calculate basic price features
    df['returns'] = df['close'].pct_change()
    df['intraday_return'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    df['overnight_return'] = (df['close'] - df['close'].shift(1)) / (df['high'] - df['low']).replace(0, np.nan)
    
    # Volatility-Normalized Intraday Momentum
    # Rolling volatility (5-day)
    df['volatility_5d'] = df['returns'].rolling(window=5).std()
    # Combined intraday momentum
    df['intraday_momentum'] = (df['intraday_return'] + df['overnight_return']) / 2
    # Volatility normalization
    df['vol_norm_momentum'] = df['intraday_momentum'] / df['volatility_5d'].replace(0, np.nan)
    
    # Multi-Timeframe Volume Confirmation
    # Short-term volume momentum (3-day)
    df['volume_3d_momentum'] = df['volume'] / df['volume'].shift(3) - 1
    df['volume_3d_trend'] = df['volume'].rolling(window=3).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    
    # Medium-term volume regime (10-day)
    df['volume_10d_avg'] = df['volume'].rolling(window=10).mean()
    df['volume_regime'] = df['volume'] / df['volume_10d_avg'].replace(0, np.nan)
    df['above_avg_volume_count'] = df['volume'].rolling(window=10).apply(lambda x: sum(x > x.mean()))
    
    # Volume-Price Alignment
    df['price_change'] = df['close'].pct_change()
    df['abs_price_change'] = df['price_change'].abs()
    
    # 10-day volume-price correlation
    volume_price_corr = []
    for i in range(len(df)):
        if i >= 9:
            window_volume = df['volume'].iloc[i-9:i+1]
            window_abs_change = df['abs_price_change'].iloc[i-9:i+1]
            if len(window_volume) == 10 and len(window_abs_change) == 10:
                corr = window_volume.corr(window_abs_change)
                volume_price_corr.append(corr if not np.isnan(corr) else 0)
            else:
                volume_price_corr.append(0)
        else:
            volume_price_corr.append(0)
    df['volume_price_corr'] = volume_price_corr
    
    # Direction consistency
    df['price_direction'] = np.sign(df['price_change'])
    df['volume_direction'] = np.sign(df['volume'] - df['volume'].shift(1))
    df['direction_consistency'] = (df['price_direction'] == df['volume_direction']).astype(int)
    
    # Adaptive Signal Combination
    # Multi-timeframe volume convergence
    short_term_volume = (df['volume_3d_momentum'] + df['volume_3d_trend']) / 2
    medium_term_volume = (df['volume_regime'] + df['above_avg_volume_count'] / 10) / 2
    
    # Weighted average with alignment bonus
    volume_convergence = 0.4 * short_term_volume + 0.6 * medium_term_volume
    alignment_bonus = df['direction_consistency'] * 0.2
    volume_score = volume_convergence + alignment_bonus
    
    # Volume-Price alignment multiplier
    alignment_multiplier = 1 + df['volume_price_corr'] * df['direction_consistency']
    
    # Recent factor effectiveness (15-day rolling correlation with future returns)
    future_returns = df['returns'].shift(-1)
    effectiveness = []
    for i in range(len(df)):
        if i >= 14:
            window_momentum = df['vol_norm_momentum'].iloc[i-14:i+1]
            window_future = future_returns.iloc[i-14:i+1]
            if len(window_momentum) == 15 and len(window_future) == 15:
                corr = window_momentum.corr(window_future)
                effectiveness.append(corr if not np.isnan(corr) else 1)
            else:
                effectiveness.append(1)
        else:
            effectiveness.append(1)
    df['effectiveness'] = effectiveness
    
    # Final alpha factor
    alpha = df['vol_norm_momentum'] * volume_score * alignment_multiplier * df['effectiveness']
    
    # Clean and return
    alpha = alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    return alpha
