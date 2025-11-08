import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining price momentum, volatility regime adjustment,
    liquidity-weighted reversal, intraday persistence, and multi-timeframe convergence.
    """
    # Create copy to avoid modifying original dataframe
    data = df.copy()
    
    # 1. Price Momentum with Volume Confirmation
    # Short-term momentum (5-day return)
    mom_5d = data['close'].pct_change(periods=5)
    
    # Medium-term momentum (20-day return)
    mom_20d = data['close'].pct_change(periods=20)
    
    # Volume trend components
    vol_ma_10 = data['volume'].rolling(window=10).mean()
    vol_ma_20 = data['volume'].rolling(window=20).mean()
    
    # Recent volume vs historical average
    vol_ratio_10 = data['volume'] / vol_ma_10
    vol_ratio_20 = data['volume'] / vol_ma_20
    
    # Volume acceleration (rate of change)
    vol_accel = data['volume'].pct_change(periods=5)
    
    # Combined momentum-volume signal
    momentum_volume = (mom_5d * 0.4 + mom_20d * 0.6) * (vol_ratio_10 * 0.5 + vol_ratio_20 * 0.5 + vol_accel * 0.2)
    
    # 2. Volatility Regime Adjusted Strength
    # Normalized daily range
    daily_range = (data['high'] - data['low']) / data['close']
    range_ma_10 = daily_range.rolling(window=10).mean()
    
    # Volatility regime identification
    returns = data['close'].pct_change()
    vol_20d = returns.rolling(window=20).std()
    vol_regime = vol_20d / vol_20d.rolling(window=60).mean()
    
    # Price strength adjusted for volatility regime
    strength_signal = (daily_range / range_ma_10) * np.where(vol_regime > 1.2, 0.7, 
                                                           np.where(vol_regime < 0.8, 1.3, 1.0))
    
    # 3. Liquidity-Weighted Reversal Signal
    # Short-term reversal components
    ret_1d = data['close'].pct_change(periods=1)
    ret_2d = data['close'].pct_change(periods=2)
    
    # Oversold/overbought conditions
    rsi_5 = 100 - (100 / (1 + (data['close'].pct_change(periods=1).rolling(window=5).mean().clip(lower=0.0001) / 
                              -data['close'].pct_change(periods=1).rolling(window=5).mean().clip(upper=-0.0001))))
    
    reversal_signal = -ret_1d * 0.6 - ret_2d * 0.4 + (50 - rsi_5) * 0.02
    
    # Liquidity measures
    dollar_volume = data['volume'] * data['close']
    dollar_vol_ma_10 = dollar_volume.rolling(window=10).mean()
    volume_consistency = data['volume'].rolling(window=10).std() / data['volume'].rolling(window=10).mean()
    
    # Liquidity-weighted reversal
    liquidity_weight = (dollar_volume / dollar_vol_ma_10) * (1 / (1 + volume_consistency))
    weighted_reversal = reversal_signal * liquidity_weight
    
    # 4. Intraday Persistence Factor
    # Opening gap analysis
    gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    gap_fill_ratio = (data['close'] - data['open']) / (data['close'].shift(1) - data['open']).replace(0, 0.0001)
    
    # Volume profile analysis
    # Assuming volume is daily volume, we simulate intraday concentration
    # by comparing recent volume patterns
    early_vol_ratio = data['volume'].rolling(window=5).apply(lambda x: x[-1] / x.mean() if x.mean() > 0 else 1)
    
    persistence_signal = gap * (1 - abs(gap_fill_ratio)) * early_vol_ratio
    
    # 5. Multi-timeframe Momentum Convergence
    # Short-term acceleration
    mom_5d_roc = mom_5d.diff(periods=3)  # Rate of change of 5-day momentum
    price_accel = data['close'].pct_change(periods=1).diff(periods=2)  # Second derivative
    
    # Medium-term trend quality
    high_ma_10 = data['high'].rolling(window=10).mean()
    low_ma_10 = data['low'].rolling(window=10).mean()
    close_ma_10 = data['close'].rolling(window=10).mean()
    
    trend_smoothness = 1 - (data['close'].rolling(window=10).std() / close_ma_10)
    pullback_resilience = (data['close'] - low_ma_10) / (high_ma_10 - low_ma_10).replace(0, 0.0001)
    
    convergence_signal = (mom_5d_roc * 0.4 + price_accel * 0.3) * (trend_smoothness * 0.5 + pullback_resilience * 0.5)
    
    # Combine all factors with appropriate weights
    final_factor = (
        momentum_volume.rank(pct=True) * 0.25 +
        strength_signal.rank(pct=True) * 0.20 +
        weighted_reversal.rank(pct=True) * 0.25 +
        persistence_signal.rank(pct=True) * 0.15 +
        convergence_signal.rank(pct=True) * 0.15
    )
    
    # Handle NaN values
    final_factor = final_factor.fillna(0)
    
    return final_factor
