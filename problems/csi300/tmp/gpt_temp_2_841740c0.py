import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Generate alpha factor combining multiple technical heuristics
    """
    df = data.copy()
    
    # 1. Price Momentum with Volume Confirmation
    momentum_5d = df['close'].pct_change(5)
    vol_ma_20d = df['volume'].rolling(20).mean()
    vol_confirmation = df['volume'] / vol_ma_20d
    momentum_signal = momentum_5d * np.where(vol_confirmation > 1, 1, 0.5)
    
    # 2. Volume-Weighted Price Reversal
    mean_reversion_3d = -df['close'].pct_change(3)
    vol_percentile = df['volume'].rolling(20).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.7)).astype(float), raw=False
    )
    reversal_signal = mean_reversion_3d * vol_percentile
    
    # 3. Intraday Range Efficiency
    daily_range = (df['high'] - df['low']) / df['close'].shift(1)
    range_per_volume = daily_range / (df['volume'] + 1e-6)
    efficiency_signal = -range_per_volume  # Negative because high efficiency = lower values
    
    # 4. Pressure Accumulation Signal
    daily_pressure = (df['close'] - df['open']) / df['open']
    above_avg_vol = (df['volume'] > df['volume'].rolling(20).mean()).astype(float)
    pressure_accumulation = (daily_pressure * above_avg_vol).rolling(5).sum()
    
    # 5. Trend Consistency Factor
    sma_3d = df['close'].rolling(3).mean()
    sma_10d = df['close'].rolling(10).mean()
    price_trend_alignment = np.sign(sma_3d.diff()) * np.sign(sma_10d.diff())
    vol_trend = np.sign(df['volume'].rolling(5).mean().diff())
    trend_consistency = price_trend_alignment * vol_trend
    
    # 6. Opening Gap Follow-Through
    opening_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    intraday_high = (df['high'] - df['open']) / df['open']
    intraday_low = (df['low'] - df['open']) / df['open']
    gap_persistence = np.where(
        opening_gap > 0, 
        (df['close'] - df['open']) / (df['high'] - df['open'] + 1e-6),
        (df['close'] - df['open']) / (df['open'] - df['low'] + 1e-6)
    )
    gap_signal = opening_gap * gap_persistence
    
    # 7. Liquidity-Adjusted Momentum
    price_momentum_5d = df['close'].pct_change(5)
    avg_trade_size = df['amount'] / (df['volume'] + 1e-6)
    liquidity_adj = 1 / (avg_trade_size.rolling(5).std() + 1e-6)
    liquidity_momentum = price_momentum_5d * liquidity_adj
    
    # 8. Volatility-Regime Momentum
    atr = (df['high'] - df['low']).rolling(10).mean()
    vol_regime = 1 / (atr / df['close'] + 1e-6)
    vol_adjusted_momentum = momentum_5d * vol_regime
    
    # 9. Volume-Price Divergence
    price_direction_3d = np.sign(df['close'].pct_change(3))
    vol_direction_3d = np.sign(df['volume'].pct_change(3))
    divergence = np.where(
        price_direction_3d != vol_direction_3d,
        -price_direction_3d,
        price_direction_3d
    )
    
    # 10. Cumulative Buying Pressure
    buying_pressure = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-6)
    positive_pressure = np.where(buying_pressure > 0.5, buying_pressure, 0)
    cumulative_pressure = positive_pressure.rolling(5).sum()
    
    # Combine all signals with equal weights
    signals = pd.DataFrame({
        'momentum': momentum_signal,
        'reversal': reversal_signal,
        'efficiency': efficiency_signal,
        'pressure': pressure_accumulation,
        'trend': trend_consistency,
        'gap': gap_signal,
        'liquidity': liquidity_momentum,
        'volatility': vol_adjusted_momentum,
        'divergence': divergence,
        'buying': cumulative_pressure
    })
    
    # Z-score normalize each component and combine
    normalized_signals = signals.apply(lambda x: (x - x.mean()) / x.std())
    combined_factor = normalized_signals.mean(axis=1)
    
    return combined_factor
