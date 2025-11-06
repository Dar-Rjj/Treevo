import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a composite alpha factor combining price-volume divergence, 
    volatility-regime adaptation, liquidity-enhanced momentum, gap mean reversion,
    and intraday session persistence.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Price-Volume Divergence Factor
    # Price momentum components
    data['high_5d_slope'] = (data['high'] - data['high'].shift(5)) / data['high'].shift(5)
    data['low_20d_slope'] = (data['low'] - data['low'].shift(20)) / data['low'].shift(20)
    price_momentum = (data['high_5d_slope'] + data['low_20d_slope']) / 2
    
    # Volume divergence component
    data['volume_10d_slope'] = (data['volume'] - data['volume'].shift(10)) / data['volume'].shift(10)
    
    # Combined divergence signal (avoid division by zero)
    price_denom = np.where(np.abs(price_momentum) > 1e-8, price_momentum, 1e-8)
    divergence_signal = data['volume_10d_slope'] / price_denom * np.sign(price_momentum)
    price_volume_divergence = divergence_signal.rolling(window=5, min_periods=3).mean()
    
    # 2. Volatility-Regime Adaptive Factor
    # Volatility regime detection
    data['ret_20d_std'] = data['close'].pct_change().rolling(window=20, min_periods=15).std()
    data['vol_60d_median'] = data['ret_20d_std'].rolling(window=60, min_periods=40).median()
    regime_indicator = data['ret_20d_std'] / data['vol_60d_median']
    
    # Pattern recognition
    data['range_3d'] = (data['high'].rolling(window=3, min_periods=2).max() - 
                        data['low'].rolling(window=3, min_periods=2).min()) / data['close']
    data['range_10d'] = (data['high'].rolling(window=10, min_periods=7).max() - 
                         data['low'].rolling(window=10, min_periods=7).min()) / data['close']
    reversal_pattern = data['range_3d'] / data['range_10d']
    
    # Breakout patterns
    data['high_20d'] = data['high'].rolling(window=20, min_periods=15).max()
    data['low_20d'] = data['low'].rolling(window=20, min_periods=15).min()
    breakout_pattern = (data['close'] - data['low_20d']) / (data['high_20d'] - data['low_20d'] + 1e-8)
    
    volatility_adaptive = regime_indicator * (reversal_pattern - breakout_pattern)
    
    # 3. Liquidity-Enhanced Momentum
    # Raw momentum signals
    data['mom_5d'] = data['close'].pct_change(periods=5)
    data['mom_10d'] = data['close'].pct_change(periods=10)
    raw_momentum = (data['mom_5d'] + data['mom_10d']) / 2
    
    # Liquidity adjustment
    data['liq_ratio'] = data['volume'] / (data['amount'] + 1e-8)
    liq_autocorr = data['liq_ratio'].rolling(window=8, min_periods=5).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
    )
    liquidity_enhanced = raw_momentum * data['liq_ratio'] * (1 + liq_autocorr)
    
    # 4. Opening Gap Mean Reversion
    # Gap measurement
    data['abs_gap'] = data['open'] - data['close'].shift(1)
    data['rel_gap'] = data['abs_gap'] / data['close'].shift(1)
    
    # Volume confirmation (using opening hour proxy - first 30min volume intensity)
    # Assuming we don't have intraday data, use daily volume as proxy
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    volume_surprise = data['volume'] / data['volume_5d_avg']
    
    gap_mean_reversion = -data['rel_gap'] * volume_surprise
    
    # 5. Intraday Session Persistence (using daily data as proxy)
    # Session strength analysis - using price movement patterns
    # Morning proxy: first half of day return (Open to (High+Low)/2)
    data['morning_perf'] = ((data['high'] + data['low']) / 2 - data['open']) / data['open']
    # Afternoon proxy: second half of day return ((High+Low)/2 to Close)
    data['afternoon_perf'] = (data['close'] - (data['high'] + data['low']) / 2) / ((data['high'] + data['low']) / 2)
    
    # Session correlation
    session_corr = data['morning_perf'].rolling(window=10, min_periods=7).corr(data['afternoon_perf'])
    volume_weighted_persistence = (data['morning_perf'] + data['afternoon_perf']) * session_corr * data['volume']
    
    intraday_persistence = volume_weighted_persistence.rolling(window=5, min_periods=3).mean()
    
    # Combine all factors with equal weights
    factors = pd.DataFrame({
        'price_volume_div': price_volume_divergence,
        'vol_adaptive': volatility_adaptive,
        'liq_momentum': liquidity_enhanced,
        'gap_reversion': gap_mean_reversion,
        'intraday_persist': intraday_persistence
    })
    
    # Z-score normalization for each factor
    normalized_factors = factors.apply(lambda x: (x - x.rolling(window=60, min_periods=40).mean()) / 
                                     x.rolling(window=60, min_periods=40).std())
    
    # Equal-weighted composite factor
    composite_factor = normalized_factors.mean(axis=1)
    
    return composite_factor
