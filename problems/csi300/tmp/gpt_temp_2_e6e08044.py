import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Weighted Price Acceleration
    # Calculate Price Acceleration
    returns = df['close'].pct_change()
    accel = returns.diff()  # Second derivative approximation
    
    # Triple Exponential Smoothing
    alpha = 0.3
    single_smooth = accel.ewm(alpha=alpha, adjust=False).mean()
    double_smooth = single_smooth.ewm(alpha=alpha, adjust=False).mean()
    triple_smooth = double_smooth.ewm(alpha=alpha, adjust=False).mean()
    smoothed_accel = triple_smooth
    
    # Volume Weighting Scheme
    vol_rank = df['volume'].rolling(window=50).apply(
        lambda x: (x.rank(pct=True).iloc[-1]), raw=False
    )
    vol_weight = 1 / (1 + np.exp(-5 * (vol_rank - 0.5)))  # Sigmoid transformation
    
    # Combine Acceleration and Volume
    accel_weighted = smoothed_accel * vol_weight
    
    # Momentum Persistence Filter
    persistence = np.zeros(len(accel_weighted))
    current_streak = 0
    for i in range(1, len(accel_weighted)):
        if (accel_weighted.iloc[i] > 0 and accel_weighted.iloc[i-1] > 0) or \
           (accel_weighted.iloc[i] < 0 and accel_weighted.iloc[i-1] < 0):
            current_streak += 1
        else:
            current_streak = 0
        persistence[i] = current_streak
    
    accel_weighted_persistent = accel_weighted * (1 + 0.1 * persistence)
    
    # Bid-Ask Spread Momentum Divergence
    spread_proxy = (df['high'] - df['low']) / df['close']
    spread_momentum = spread_proxy.diff(5)
    price_momentum = df['close'].pct_change(5)
    
    # Detect Divergence Patterns
    price_volatility = df['close'].pct_change().rolling(window=5).std()
    low_vol_periods = price_volatility < price_volatility.rolling(window=20).quantile(0.3)
    high_move_periods = price_momentum.abs() > price_momentum.abs().rolling(window=20).quantile(0.7)
    
    bearish_div = (spread_momentum > 0) & low_vol_periods
    bullish_div = (spread_momentum < 0) & high_move_periods
    
    divergence_signal = np.zeros(len(df))
    divergence_signal[bearish_div] = -spread_momentum[bearish_div] * price_momentum[bearish_div].abs()
    divergence_signal[bullish_div] = -spread_momentum[bullish_div] * price_momentum[bullish_div]
    
    # Overnight Gap Mean Reversion
    overnight_ret = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Gap Persistence
    gap_direction = np.sign(overnight_ret)
    gap_persistence = gap_direction.rolling(window=3).apply(
        lambda x: len(set(x)) == 1 and not pd.isna(x).any(), raw=False
    ).astype(float)
    
    # Gap Reversion Probability
    def calc_reversion_prob(window):
        if len(window) < 2:
            return 0.5
        reversions = 0
        for i in range(1, len(window)):
            if window[i] * window[i-1] < 0:  # Opposite signs
                reversions += 1
        return reversions / (len(window) - 1)
    
    reversion_prob = overnight_ret.rolling(window=60).apply(calc_reversion_prob, raw=False)
    
    # Volume Confirmation
    vol_zscore = (df['volume'] - df['volume'].rolling(window=20).mean()) / df['volume'].rolling(window=20).std()
    vol_confirmation = (vol_zscore > 1).astype(float)
    
    gap_signal = -overnight_ret * reversion_prob * (1 + 0.5 * vol_confirmation)
    
    # Combine all factors
    combined_factor = (
        0.4 * accel_weighted_persistent + 
        0.3 * divergence_signal + 
        0.3 * gap_signal
    )
    
    return pd.Series(combined_factor, index=df.index)
