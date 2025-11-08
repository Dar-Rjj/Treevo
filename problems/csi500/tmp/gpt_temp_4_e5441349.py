import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining intraday momentum decay with volume acceleration,
    volatility-scaled momentum, liquidity-efficient reversal, regime-aware order flow, 
    and multi-timeframe breakout confidence.
    """
    # Copy data to avoid modifying original
    data = df.copy()
    
    # 1. Intraday Momentum Decay Factor
    # Calculate intraday momentum
    mid_price = (data['high'] + data['low']) / 2
    prev_close = data['close'].shift(1)
    intraday_momentum = (mid_price - prev_close) / prev_close
    
    # Compute momentum decay with volatility scaling
    price_range = (data['high'] - data['low']) / prev_close
    
    # Exponential smoothing functions
    def exp_smooth(series, half_life):
        alpha = 1 - np.exp(np.log(0.5) / half_life)
        return series.ewm(alpha=alpha).mean()
    
    # Apply exponential smoothing
    smooth_momentum = exp_smooth(intraday_momentum, 3)
    smooth_volatility = exp_smooth(price_range, 5)
    decay_adjusted_momentum = smooth_momentum / (1 + smooth_volatility)
    
    # Volume acceleration
    volume_accel = (data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)
    
    # Combine with volume acceleration
    momentum_volume = decay_adjusted_momentum * volume_accel
    
    # 2. Volatility-Scaled Momentum with Volume Confirmation
    # Multi-timeframe momentum
    short_momentum = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    medium_momentum = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    
    smooth_short = exp_smooth(short_momentum, 2)
    smooth_medium = exp_smooth(medium_momentum, 5)
    
    # Blend timeframes
    blended_momentum = 0.6 * smooth_short + 0.4 * smooth_medium
    
    # Adaptive volatility scaling
    high_vol_regime = data['high'].rolling(5).max() - data['low'].rolling(5).min()
    low_vol_regime = data['high'].rolling(10).max() - data['low'].rolling(10).min()
    
    # Regime detection using rolling median of volatility
    vol_median = price_range.rolling(20).median()
    regime_vol = np.where(price_range > vol_median, high_vol_regime, low_vol_regime)
    
    volatility_scaled_momentum = blended_momentum / (regime_vol + 1e-8)
    
    # Volume confirmation
    volume_momentum = (data['volume'] - data['volume'].shift(3)) / data['volume'].shift(3)
    smooth_volume_mom = exp_smooth(volume_momentum, 3)
    
    # Volume acceleration threshold and persistence
    vol_threshold = smooth_volume_mom.rolling(5).quantile(0.7)
    vol_confirmation = ((smooth_volume_mom > vol_threshold) & 
                       (smooth_volume_mom.shift(1) > vol_threshold)).astype(float)
    
    # 3. Liquidity-Efficient Price Reversal
    # Price deviation from 5-day MA
    ma_5 = data['close'].rolling(5).mean()
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    price_deviation = (data['close'] - ma_5) / typical_price
    
    # Extreme moves detection
    extreme_threshold = price_deviation.rolling(20).quantile(0.9)
    overreaction = np.abs(price_deviation) * (np.abs(price_deviation) > extreme_threshold)
    
    # Liquidity conditions
    price_range_daily = data['high'] - data['low']
    amount_efficiency = data['amount'] / (price_range_daily + 1e-8)
    liquidity_ratio = amount_efficiency / amount_efficiency.rolling(10).mean()
    
    volume_concentration = data['volume'] / data['volume'].rolling(5).mean()
    liquidity_score = exp_smooth(liquidity_ratio * volume_concentration, 5)
    
    # Reversal signal
    reversal_signal = -overreaction * liquidity_score
    
    # 4. Regime-Aware Order Flow Momentum
    # Directional amount flow
    up_days = (data['close'] > data['open']).astype(int)
    down_days = (data['close'] < data['open']).astype(int)
    directional_amount = data['amount'] * (up_days - down_days)
    
    # Net order flow
    net_order_flow = directional_amount.rolling(3).sum() / data['amount'].rolling(3).sum()
    smooth_order_flow = exp_smooth(net_order_flow, 5)
    
    # Regime detection
    vol_regime_10d = (data['high'].rolling(10).max() - data['low'].rolling(10).min()) / data['close']
    regime_percentile = vol_regime_10d.rolling(20).rank(pct=True)
    
    # Adaptive lookback based on regime
    regime_lookback = np.where(regime_percentile > 0.7, 3, 7)
    regime_weighted_flow = smooth_order_flow.rolling(window=5).apply(
        lambda x: x.iloc[-int(regime_lookback[-1]):].mean() if len(x) >= int(regime_lookback[-1]) else x.mean()
    )
    
    # 5. Multi-Timeframe Breakout Confidence
    # Normalized range
    true_range = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    normalized_range = true_range / data['close']
    
    # Breakout detection
    range_5d_percentile = normalized_range.rolling(5).rank(pct=True)
    range_10d_percentile = normalized_range.rolling(10).rank(pct=True)
    breakout_strength = (range_5d_percentile + range_10d_percentile) / 2
    
    # Volume and amount validation
    volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
    amount_efficiency_breakout = data['amount'] / (true_range + 1e-8)
    efficiency_ratio = amount_efficiency_breakout / amount_efficiency_breakout.rolling(10).mean()
    
    validation_score = (volume_ratio * efficiency_ratio).clip(0, 2)
    
    # Confidence-weighted breakout
    confidence_breakout = breakout_strength * validation_score
    
    # Combine all factors with weights
    final_alpha = (
        0.25 * momentum_volume +
        0.20 * volatility_scaled_momentum * vol_confirmation +
        0.20 * reversal_signal +
        0.20 * regime_weighted_flow +
        0.15 * confidence_breakout
    )
    
    # Final normalization
    final_alpha = (final_alpha - final_alpha.rolling(20).mean()) / final_alpha.rolling(20).std()
    
    return final_alpha
