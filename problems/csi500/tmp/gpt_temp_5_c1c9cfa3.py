import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a composite alpha factor combining multiple market insights.
    
    Parameters:
    df: DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
    
    Returns:
    Series: Composite alpha factor values indexed by date
    """
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Volatility-Adjusted Price Momentum
    # Calculate short and medium-term momentum
    ret_5d = df['close'].pct_change(5)
    ret_20d = df['close'].pct_change(20)
    
    # Calculate volatility using high-low range and return std
    hl_range = (df['high'] - df['low']) / df['close']
    vol_hl = hl_range.rolling(20).mean()
    vol_ret = df['close'].pct_change().rolling(20).std()
    
    # Combined volatility measure
    volatility = (vol_hl + vol_ret) / 2
    
    # Volatility-adjusted momentum
    vol_adj_momentum = (ret_5d - ret_20d) / (volatility + 1e-8)
    
    # Volume-Price Divergence Factor
    vol_ma_10 = df['volume'].rolling(10).mean()
    vol_roc = df['volume'] / vol_ma_10 - 1
    
    price_ma_10 = df['close'].rolling(10).mean()
    price_roc = df['close'] / price_ma_10 - 1
    
    # Calculate rolling correlation between volume and price trends
    vol_price_corr = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        window = df.iloc[i-20:i]
        corr = window['volume'].pct_change().corr(window['close'].pct_change())
        vol_price_corr.iloc[i] = corr
    
    volume_price_divergence = vol_roc - price_roc * (1 - vol_price_corr.abs())
    
    # Intraday Reversal Strength
    morning_high_ret = (df['high'] - df['open']) / df['open']
    morning_low_ret = (df['low'] - df['open']) / df['open']
    morning_pressure = (morning_high_ret.abs() + morning_low_ret.abs()) / 2
    
    afternoon_high_ret = (df['close'] - df['high']) / df['high']
    afternoon_low_ret = (df['close'] - df['low']) / df['low']
    afternoon_recovery = (afternoon_high_ret.abs() + afternoon_low_ret.abs()) / 2
    
    reversal_signal = (afternoon_recovery - morning_pressure) * (df['volume'] / df['volume'].rolling(20).mean())
    
    # Liquidity-Efficient Return
    ret_10d = df['close'].pct_change(10)
    avg_volume = df['volume'].rolling(10).mean()
    price_impact = (df['high'] - df['low']) / df['close'] * df['amount']
    
    liquidity_efficiency = ret_10d / (price_impact / avg_volume + 1e-8)
    
    # Regime-Adaptive Trend Following
    vol_regime = df['close'].pct_change().rolling(20).std()
    high_vol = vol_regime > vol_regime.rolling(50).quantile(0.7)
    
    ma_slope = (df['close'].rolling(20).mean() - df['close'].rolling(50).mean()) / df['close'].rolling(50).mean()
    trending = ma_slope.abs() > ma_slope.rolling(50).quantile(0.7)
    
    short_momentum = df['close'].pct_change(5)
    long_momentum = df['close'].pct_change(20)
    
    regime_adaptive = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if high_vol.iloc[i]:
            regime_adaptive.iloc[i] = short_momentum.iloc[i]
        else:
            regime_adaptive.iloc[i] = long_momentum.iloc[i]
        
        if trending.iloc[i]:
            regime_adaptive.iloc[i] *= 1.5
    
    # Order Flow Imbalance
    up_days = df['close'] > df['open']
    down_days = df['close'] < df['open']
    
    buy_volume = df['volume'].where(up_days, 0).rolling(10).sum()
    sell_volume = df['volume'].where(down_days, 0).rolling(10).sum()
    
    buy_pressure = (buy_volume * (df['close'] - df['open']).where(up_days, 0)).rolling(10).sum()
    sell_pressure = (sell_volume * (df['open'] - df['close']).where(down_days, 0)).rolling(10).sum()
    
    order_flow_imbalance = (buy_pressure - sell_pressure) / (buy_volume + sell_volume + 1e-8)
    
    # Gap Filling Probability
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Simplified gap filling detection
    gap_filled = pd.Series(index=df.index, dtype=bool)
    for i in range(1, len(df)):
        if overnight_gap.iloc[i] > 0:  # Gap up
            gap_filled.iloc[i] = df['low'].iloc[i] <= df['close'].shift(1).iloc[i]
        else:  # Gap down
            gap_filled.iloc[i] = df['high'].iloc[i] >= df['close'].shift(1).iloc[i]
    
    gap_fill_prob = gap_filled.rolling(50).mean()
    gap_signal = overnight_gap * (1 - gap_fill_prob) * (df['volume'] / df['volume'].rolling(20).mean())
    
    # Multi-Timeframe Momentum Convergence
    mom_short = df['close'].pct_change(3)
    mom_medium = df['close'].pct_change(10) - df['close'].pct_change(20)
    mom_long = df['close'].pct_change(30) - df['close'].pct_change(60)
    
    # Calculate convergence score
    convergence_score = (np.sign(mom_short) + np.sign(mom_medium) + np.sign(mom_long)) / 3
    
    # Combine all factors with equal weights
    factors = [
        vol_adj_momentum,
        volume_price_divergence,
        reversal_signal,
        liquidity_efficiency,
        regime_adaptive,
        order_flow_imbalance,
        gap_signal,
        convergence_score
    ]
    
    # Normalize each factor and combine
    combined_factor = pd.Series(0, index=df.index)
    for factor in factors:
        normalized = (factor - factor.rolling(100).mean()) / (factor.rolling(100).std() + 1e-8)
        combined_factor += normalized
    
    # Final normalization
    result = (combined_factor - combined_factor.rolling(100).mean()) / (combined_factor.rolling(100).std() + 1e-8)
    
    return result
