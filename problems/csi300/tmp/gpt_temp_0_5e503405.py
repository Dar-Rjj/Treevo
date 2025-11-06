import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining volatility-regime breakout, intraday momentum reversal,
    gap filling probability, and liquidity-adjusted trend strength.
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility-Regime Range Breakout Component
    # 5-day vs 20-day volatility ratio
    vol_5d = data['high'].rolling(window=5).std() / data['low'].rolling(window=5).mean()
    vol_20d = data['high'].rolling(window=20).std() / data['low'].rolling(window=20).mean()
    vol_ratio = vol_5d / vol_20d
    
    # Dynamic breakout threshold based on volatility regime
    high_low_range = (data['high'] - data['low']) / data['close']
    scaled_range = high_low_range * vol_ratio
    
    # Volume-price confirmation
    volume_20d_avg = data['volume'].rolling(window=20).mean()
    volume_spike = data['volume'] / volume_20d_avg
    regime_volume_alignment = volume_spike * vol_ratio
    
    breakout_component = scaled_range * regime_volume_alignment
    
    # 2. Intraday Momentum Reversal Component
    # First hour momentum (assuming first hour is represented by early data)
    open_to_high = (data['high'] - data['open']) / data['open']
    open_to_low = (data['low'] - data['open']) / data['open']
    first_hour_strength = np.where(open_to_high > abs(open_to_low), open_to_high, open_to_low)
    
    # Late session reversal (assuming last hour represented by close vs intraday extremes)
    high_to_close = (data['close'] - data['high']) / data['high']
    low_to_close = (data['close'] - data['low']) / data['low']
    late_reversal = np.where(high_to_close < low_to_close, high_to_close, low_to_close)
    
    # Bid-ask spread pressure proxy using high-low range vs close
    spread_pressure = (data['high'] - data['low']) / data['close']
    
    # Morning-afternoon volume ratio (approximated)
    morning_volume = data['volume'].rolling(window=5).apply(lambda x: x.iloc[0] if len(x) == 5 else np.nan)
    afternoon_volume = data['volume'].rolling(window=5).apply(lambda x: x.iloc[-1] if len(x) == 5 else np.nan)
    volume_ratio = morning_volume / afternoon_volume
    
    momentum_component = first_hour_strength * late_reversal * spread_pressure * volume_ratio
    
    # 3. Gap Filling Probability Component
    # Overnight gap (previous close to current open)
    overnight_gap = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Intraday fade validation
    gap_fill_progress = np.where(overnight_gap > 0, 
                                (data['low'] - data['open']) / overnight_gap,
                                (data['high'] - data['open']) / overnight_gap)
    
    # Early session volume concentration (first 25% of day volume proxy)
    total_volume = data['volume']
    early_volume_ratio = morning_volume / total_volume
    
    gap_component = overnight_gap * gap_fill_progress * early_volume_ratio
    
    # 4. Liquidity-Adjusted Trend Strength Component
    # Decay-weighted momentum with recent period emphasis
    price_changes = data['close'].pct_change()
    
    # Multi-period weighted momentum (5, 10, 20 days with decay)
    momentum_5d = price_changes.rolling(window=5).mean()
    momentum_10d = price_changes.rolling(window=10).mean()
    momentum_20d = price_changes.rolling(window=20).mean()
    
    decay_weighted_momentum = (0.5 * momentum_5d + 0.3 * momentum_10d + 0.2 * momentum_20d)
    
    # Volume-price divergence
    volume_trend = data['volume'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan
    )
    price_trend = data['close'].rolling(window=10).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else np.nan
    )
    
    trend_divergence = np.sign(price_trend) * np.sign(volume_trend) * abs(price_trend / (abs(volume_trend) + 1e-8))
    
    trend_component = decay_weighted_momentum * trend_divergence
    
    # Combine all components with appropriate weights
    final_factor = (
        0.3 * breakout_component.fillna(0) +
        0.25 * momentum_component.fillna(0) +
        0.2 * gap_component.fillna(0) +
        0.25 * trend_component.fillna(0)
    )
    
    return final_factor
