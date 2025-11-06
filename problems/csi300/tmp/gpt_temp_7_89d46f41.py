import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Volatility-Weighted Momentum factor
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # 1. Regime Identification
    # True Range calculation
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility regime (20-day rolling True Range)
    data['volatility_regime'] = data['true_range'].rolling(window=20).mean()
    
    # Price fractal dimension approximation (10-day)
    def fractal_dimension(high, low):
        n = len(high)
        if n < 2:
            return np.nan
        range_sum = sum(abs(high[i] - low[i]) for i in range(n))
        price_range = max(high) - min(low)
        if price_range == 0:
            return 1.0
        return 2 - (np.log(range_sum / price_range) / np.log(n))
    
    data['fractal_dim'] = data['high'].rolling(window=10).apply(
        lambda x: fractal_dimension(x, data.loc[x.index, 'low']), raw=False
    )
    
    # Trending vs ranging regime
    data['trending_regime'] = data['fractal_dim'] < 1.5
    
    # 2. Raw Momentum Calculation
    data['ret_5d'] = data['close'] / data['close'].shift(5) - 1
    data['ret_10d'] = data['close'] / data['close'].shift(10) - 1
    data['ret_20d'] = data['close'] / data['close'].shift(20) - 1
    
    # Momentum acceleration (5-day ROC of 10-day return)
    data['mom_acceleration'] = (data['ret_10d'] - data['ret_10d'].shift(5)) / abs(data['ret_10d'].shift(5)).replace(0, np.nan)
    
    # Combined momentum score
    data['raw_momentum'] = (data['ret_5d'] * 0.4 + data['ret_10d'] * 0.35 + data['ret_20d'] * 0.25)
    
    # 3. Volatility Adjustment
    data['daily_range'] = data['high'] - data['low']
    data['avg_range_20d'] = data['daily_range'].rolling(window=20).mean()
    
    # Asymmetric volatility weighting
    data['volatility_weight'] = 1.0
    
    # Separate positive and negative return days
    positive_ret_mask = data['raw_momentum'] > 0
    negative_ret_mask = data['raw_momentum'] < 0
    
    # Higher weight for low volatility down days
    low_vol_down = negative_ret_mask & (data['daily_range'] < data['avg_range_20d'])
    data.loc[low_vol_down, 'volatility_weight'] = 1.5
    
    # Lower weight for high volatility up days
    high_vol_up = positive_ret_mask & (data['daily_range'] > data['avg_range_20d'])
    data.loc[high_vol_up, 'volatility_weight'] = 0.7
    
    # Volatility-adjusted momentum
    data['vol_adj_momentum'] = data['raw_momentum'] * data['volatility_weight']
    
    # 4. Divergence Detection
    # Price-volume trend correlation (10-day)
    data['price_trend'] = data['close'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    data['volume_trend'] = data['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    
    # Price-volume divergence
    data['pv_divergence'] = abs(data['price_trend'] - data['volume_trend'])
    
    # 5. Signal Generation
    data['factor'] = 0.0
    
    # High volatility regime: mean reversion with volume confirmation
    high_vol_regime = data['volatility_regime'] > data['volatility_regime'].rolling(window=50).quantile(0.7)
    
    # Negative momentum with high volume clusters
    high_volume = data['volume'] > data['volume'].rolling(window=20).quantile(0.8)
    neg_momentum_high_vol = high_vol_regime & (data['vol_adj_momentum'] < 0) & high_volume
    
    # Strong price-volume divergence in high volatility
    strong_divergence = data['pv_divergence'] > data['pv_divergence'].rolling(window=20).quantile(0.8)
    
    # High volatility signal: mean reversion
    high_vol_signal = neg_momentum_high_vol & strong_divergence
    data.loc[high_vol_signal, 'factor'] = -data.loc[high_vol_signal, 'vol_adj_momentum'] * 1.2
    
    # Low volatility regime: momentum persistence with trend strength
    low_vol_regime = data['volatility_regime'] < data['volatility_regime'].rolling(window=50).quantile(0.3)
    
    # Positive momentum with low volatility adjustment
    pos_momentum_low_vol = low_vol_regime & (data['vol_adj_momentum'] > 0)
    
    # Consistent price-volume alignment
    consistent_alignment = data['pv_divergence'] < data['pv_divergence'].rolling(window=20).quantile(0.3)
    
    # Low volatility signal: momentum persistence
    low_vol_signal = pos_momentum_low_vol & consistent_alignment & data['trending_regime']
    data.loc[low_vol_signal, 'factor'] = data.loc[low_vol_signal, 'vol_adj_momentum'] * 1.5
    
    # Neutral regime: use volatility-adjusted momentum with divergence penalty
    neutral_mask = ~high_vol_signal & ~low_vol_signal
    divergence_penalty = 1.0 - (data['pv_divergence'] / data['pv_divergence'].rolling(window=20).max()).fillna(0)
    data.loc[neutral_mask, 'factor'] = data.loc[neutral_mask, 'vol_adj_momentum'] * divergence_penalty
    
    # Clean up intermediate columns
    cols_to_drop = ['prev_close', 'tr1', 'tr2', 'tr3', 'true_range', 'volatility_regime', 
                   'fractal_dim', 'trending_regime', 'ret_5d', 'ret_10d', 'ret_20d',
                   'mom_acceleration', 'raw_momentum', 'daily_range', 'avg_range_20d',
                   'volatility_weight', 'vol_adj_momentum', 'price_trend', 'volume_trend',
                   'pv_divergence']
    
    data = data.drop(columns=cols_to_drop)
    
    return data['factor']
