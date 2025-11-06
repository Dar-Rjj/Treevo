import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Momentum Acceleration with Volume Confirmation
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum with Exponential Decay
    # Calculate rolling returns for different timeframes
    M5 = data['close'].pct_change(5)
    M10 = data['close'].pct_change(10)
    M20 = data['close'].pct_change(20)
    
    # Apply exponential decay weighting (recent momentum weighted higher)
    weights = np.array([0.5, 0.3, 0.2])  # 5-day, 10-day, 20-day weights
    decayed_momentum = (M5 * weights[0] + M10 * weights[1] + M20 * weights[2])
    
    # Momentum Acceleration
    # Calculate rate of change of decayed momentum
    mom_accel_1 = decayed_momentum - decayed_momentum.shift(1)
    mom_accel_3 = decayed_momentum - decayed_momentum.shift(3)
    mom_accel_5 = decayed_momentum - decayed_momentum.shift(5)
    
    # Apply smoothing filter (weighted average)
    momentum_acceleration = (mom_accel_1 * 0.5 + mom_accel_3 * 0.3 + mom_accel_5 * 0.2)
    momentum_acceleration = momentum_acceleration.rolling(window=5, min_periods=3).mean()
    
    # Volume Confirmation
    # Volume acceleration (ROC)
    vol_accel_1 = data['volume'].pct_change(1)
    vol_accel_3 = data['volume'].pct_change(3)
    vol_accel_5 = data['volume'].pct_change(5)
    
    # Volume trend strength
    vol_ma_ratio = data['volume'] / data['volume'].rolling(window=20, min_periods=10).mean()
    vol_consistency = data['volume'].rolling(window=10, min_periods=5).std() / data['volume'].rolling(window=10, min_periods=5).mean()
    
    # Combined volume confirmation score
    volume_trend = (vol_accel_1 * 0.4 + vol_accel_3 * 0.3 + vol_accel_5 * 0.3) * vol_ma_ratio / (1 + vol_consistency)
    
    # Market Regime Detection
    # Volatility regime
    high_low_range = (data['high'] - data['low']) / data['close']
    vol_regime_threshold = high_low_range.rolling(window=20, min_periods=10).quantile(0.7)
    high_vol_regime = (high_low_range > vol_regime_threshold).astype(int)
    
    # Trend vs Range conditions
    price_channel_high = data['high'].rolling(window=20, min_periods=10).max()
    price_channel_low = data['low'].rolling(window=20, min_periods=10).min()
    channel_position = (data['close'] - price_channel_low) / (price_channel_high - price_channel_low)
    
    trend_strength = data['close'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] / np.std(x) if len(x) > 1 else 0
    )
    trending_market = (abs(trend_strength) > trend_strength.rolling(window=20, min_periods=10).std()).astype(int)
    
    # Breakout detection
    breakout_strength = np.zeros(len(data))
    for i in range(20, len(data)):
        recent_high = data['high'].iloc[i-5:i].max()
        recent_low = data['low'].iloc[i-5:i].min()
        current_range = data['high'].iloc[i] - data['low'].iloc[i]
        
        if data['close'].iloc[i] > recent_high:
            breakout_strength[i] = (data['close'].iloc[i] - recent_high) / current_range
        elif data['close'].iloc[i] < recent_low:
            breakout_strength[i] = (data['close'].iloc[i] - recent_low) / current_range
    
    breakout_strength = pd.Series(breakout_strength, index=data.index)
    
    # Generate Adaptive Signal
    # Combine momentum acceleration with volume confirmation
    base_signal = momentum_acceleration * volume_trend
    
    # Apply regime-specific adjustments
    final_signal = np.zeros(len(data))
    
    for i in range(len(data)):
        if high_vol_regime.iloc[i] == 1:
            # High volatility regime: amplify reversal signals with shorter lookback
            vol_adjusted = base_signal.iloc[i] * (1 + 0.3 * np.sign(-base_signal.iloc[i]))
            final_signal[i] = vol_adjusted * (1 + 0.2 * breakout_strength.iloc[i])
        elif trending_market.iloc[i] == 1:
            # Trending market: enhance momentum with reduced filtering
            trend_adjusted = base_signal.iloc[i] * (1 + 0.2 * np.sign(base_signal.iloc[i]))
            final_signal[i] = trend_adjusted * (1 + 0.3 * breakout_strength.iloc[i])
        else:
            # Normal/Ranging market
            final_signal[i] = base_signal.iloc[i] * (1 + 0.1 * breakout_strength.iloc[i])
    
    # Final smoothing and normalization
    factor = pd.Series(final_signal, index=data.index)
    factor = factor.rolling(window=3, min_periods=1).mean()
    factor = (factor - factor.rolling(window=20, min_periods=10).mean()) / factor.rolling(window=20, min_periods=10).std()
    
    return factor.fillna(0)
