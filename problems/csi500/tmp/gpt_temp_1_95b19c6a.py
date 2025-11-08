import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Generate regime-sensitive alpha factor combining momentum acceleration, 
    price-volume divergence, microstructure signals, and regime transition detection.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price features
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # 1. Dynamic Regime-Sensitive Momentum Acceleration
    # Calculate momentum components
    mom_short = close / close.shift(3) - 1
    mom_medium = close / close.shift(10) - 1
    mom_acceleration = mom_medium - mom_short
    
    # Volatility regime
    daily_range = (high - low) / close
    vol_10d = daily_range.rolling(window=10).std()
    vol_20d_avg = daily_range.rolling(window=20).std()
    vol_regime = vol_10d / vol_20d_avg
    
    # Volume regime
    vol_5d_sum = volume.rolling(window=5).sum()
    vol_20d_sum = volume.rolling(window=20).sum()
    vol_ratio = vol_5d_sum / vol_20d_sum
    
    # Regime-adaptive weighting for momentum
    momentum_signal = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < 20:
            continue
            
        if vol_regime.iloc[i] > 1.2 and vol_ratio.iloc[i] > 1.1:
            # High volatility + high volume: emphasize short-term acceleration
            momentum_signal.iloc[i] = mom_acceleration.iloc[i] * 1.5 * vol_ratio.iloc[i]
        elif vol_regime.iloc[i] < 0.8 and vol_ratio.iloc[i] < 0.9:
            # Low volatility + low volume: emphasize medium-term momentum
            momentum_signal.iloc[i] = mom_medium.iloc[i] * 0.7
        else:
            # Mixed regimes: balance components
            regime_strength = abs(vol_regime.iloc[i] - 1) + abs(vol_ratio.iloc[i] - 1)
            momentum_signal.iloc[i] = (mom_acceleration.iloc[i] * 0.5 + 
                                     mom_medium.iloc[i] * 0.5) * (1 + regime_strength)
    
    # 2. Price-Volume Divergence with Regime Filtering
    # Calculate price and volume trends using linear regression
    def calc_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i < window-1:
                continue
            y = series.iloc[i-window+1:i+1].values
            x = np.arange(window)
            slope, _, _, _, _ = stats.linregress(x, y)
            slopes.iloc[i] = slope
        return slopes
    
    price_trend = calc_slope(close, 5)
    volume_trend = calc_slope(volume, 5)
    
    # Calculate divergence
    pv_divergence = price_trend * volume_trend
    
    # Regime detection
    price_changes = close.pct_change().abs()
    autocorr = price_changes.rolling(window=10).apply(
        lambda x: pd.Series(x).autocorr() if len(x) == 10 else np.nan, 
        raw=False
    )
    
    # Intraday pressure
    daily_pressure = (close - low) / (high - low).replace(0, np.nan)
    pressure_3d = daily_pressure.rolling(window=3).mean()
    
    # Apply regime filtering
    pv_signal = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < 10:
            continue
            
        trend_regime = autocorr.iloc[i] > 0.1
        pressure_confirm = 0.3 < pressure_3d.iloc[i] < 0.7
        
        if trend_regime and pressure_confirm:
            # Strong divergence in trend regimes
            pv_signal.iloc[i] = pv_divergence.iloc[i] * abs(volume_trend.iloc[i])
        elif not trend_regime and pressure_confirm:
            # Weak divergence in mean-reversion regimes
            pv_signal.iloc[i] = pv_divergence.iloc[i] * 0.3
        else:
            pv_signal.iloc[i] = pv_divergence.iloc[i] * 0.1
    
    # 3. Liquidity-Weighted Micro-Structure Alpha
    # Micro-structure signals
    daily_return_mag = (close - close.shift(1)).abs()
    intraday_efficiency = daily_return_mag / (high - low).replace(0, np.nan)
    
    vol_concentration = volume / volume.rolling(window=5).mean()
    
    # Liquidity conditions
    spread_proxy = (high - low) / close
    spread_5d_avg = spread_proxy.rolling(window=5).mean()
    
    price_level = amount / volume.replace(0, np.nan)
    price_level_10d_avg = price_level.rolling(window=10).mean()
    amount_liquidity = price_level / price_level_10d_avg
    
    # Combine with dynamic weighting
    micro_signal = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < 10:
            continue
            
        high_liquidity = (spread_proxy.iloc[i] < spread_5d_avg.iloc[i] * 0.8 and 
                         amount_liquidity.iloc[i] > 0.9)
        
        if high_liquidity:
            # Emphasize return efficiency
            micro_signal.iloc[i] = intraday_efficiency.iloc[i] * vol_concentration.iloc[i]
        else:
            # Emphasize abnormal volume patterns
            micro_signal.iloc[i] = vol_concentration.iloc[i] * 1.5 - intraday_efficiency.iloc[i] * 0.5
    
    # 4. Regime-Transition Predictive Alpha
    # Detect regime changes
    vol_5d = daily_range.rolling(window=5).std()
    vol_20d = daily_range.rolling(window=20).std()
    vol_compression = vol_5d / vol_20d
    
    vol_zscore = (volume - volume.rolling(window=20).mean()) / volume.rolling(window=20).std()
    
    # Transition signals
    mom_accel_change = mom_acceleration.diff(3)
    
    # Volume divergence during compression
    vol_price_div = vol_zscore - close.pct_change(3).abs()
    
    # Generate predictive alpha
    transition_signal = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i < 20:
            continue
            
        pre_breakout = (vol_compression.iloc[i] < 0.7 and 
                       vol_zscore.iloc[i] > 1.0)
        
        post_transition = (vol_compression.iloc[i] > 1.2 and 
                          vol_ratio.iloc[i] > 1.1)
        
        if pre_breakout:
            # Pre-breakout: volatility compression + volume buildup
            transition_prob = (0.7 - vol_compression.iloc[i]) + (vol_zscore.iloc[i] - 1.0)
            transition_signal.iloc[i] = mom_accel_change.iloc[i] * transition_prob * 2.0
        elif post_transition:
            # Post-transition: regime confirmation
            transition_signal.iloc[i] = mom_medium.iloc[i] * vol_ratio.iloc[i]
        else:
            transition_signal.iloc[i] = 0
    
    # Combine all components with dynamic weights
    for i in range(len(df)):
        if i < 20:
            result.iloc[i] = 0
            continue
        
        # Dynamic weighting based on current regime characteristics
        current_vol_regime = vol_regime.iloc[i]
        current_vol_activity = vol_ratio.iloc[i]
        
        # Weight adjustment based on market conditions
        if current_vol_regime > 1.2:
            # High volatility: emphasize momentum and transition signals
            w_momentum = 0.4
            w_pv = 0.2
            w_micro = 0.2
            w_transition = 0.2
        elif current_vol_regime < 0.8:
            # Low volatility: emphasize microstructure and price-volume signals
            w_momentum = 0.2
            w_pv = 0.3
            w_micro = 0.3
            w_transition = 0.2
        else:
            # Normal conditions: balanced approach
            w_momentum = 0.25
            w_pv = 0.25
            w_micro = 0.25
            w_transition = 0.25
        
        # Apply volume-based scaling
        vol_scale = min(max(vol_ratio.iloc[i], 0.5), 2.0)
        
        # Combine signals
        combined = (momentum_signal.iloc[i] * w_momentum +
                   pv_signal.iloc[i] * w_pv +
                   micro_signal.iloc[i] * w_micro +
                   transition_signal.iloc[i] * w_transition) * vol_scale
        
        result.iloc[i] = combined
    
    # Normalize the final signal
    result = (result - result.rolling(window=20).mean()) / result.rolling(window=20).std()
    
    return result.fillna(0)
