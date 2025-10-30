import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Initialize result series
    factor_values = pd.Series(index=data.index, dtype=float)
    
    # Calculate required intermediate series
    # Intraday momentum
    intraday_momentum = data['close'] / data['open'] - 1
    
    # Volume calculations
    volume_avg_5d = data['volume'].rolling(window=5, min_periods=1).mean()
    volume_avg_20d = data['volume'].rolling(window=20, min_periods=1).mean()
    amount_avg_5d = data['amount'].rolling(window=5, min_periods=1).mean()
    
    # Price returns for volatility calculation
    returns = data['close'].pct_change()
    returns_5d_std = returns.rolling(window=5, min_periods=1).std()
    returns_20d_std = returns.rolling(window=20, min_periods=1).std()
    
    # Price momentum calculations
    close_lag1 = data['close'].shift(1)
    close_lag5 = data['close'].shift(5)
    close_lag20 = data['close'].shift(20)
    
    # Calculate factors for each day
    for i in range(len(data)):
        if i < 1:  # Need at least 1 day of history
            factor_values.iloc[i] = 0
            continue
            
        current_date = data.index[i]
        
        # Factor 1: Momentum Decay with Volume Confirmation
        if i >= 2:
            intraday_momentum_t = intraday_momentum.iloc[i]
            intraday_momentum_t1 = intraday_momentum.iloc[i-1]
            momentum_decay = (intraday_momentum_t / intraday_momentum_t1) * np.exp(-abs(intraday_momentum_t1))
            volume_trend = data['volume'].iloc[i] / data['volume'].iloc[i-1]
            factor1 = momentum_decay * volume_trend
        else:
            factor1 = 0
        
        # Factor 2: Breakout Efficiency with Volatility Adjustment
        high_t = data['high'].iloc[i]
        low_t = data['low'].iloc[i]
        close_t = data['close'].iloc[i]
        close_t1 = close_lag1.iloc[i]
        
        true_range = max(
            high_t - low_t,
            abs(high_t - close_t1),
            abs(low_t - close_t1)
        )
        
        if true_range > 0:
            breakout_signal = (close_t - close_t1) / true_range
        else:
            breakout_signal = 0
            
        volatility_ratio = returns_5d_std.iloc[i] / returns_20d_std.iloc[i] if returns_20d_std.iloc[i] > 0 else 1
        factor2 = breakout_signal * (1 / volatility_ratio) if volatility_ratio > 0 else 0
        
        # Factor 3: Regime-Adaptive Price-Volume Divergence
        if i >= 5:
            dollar_volume_trend = data['amount'].iloc[i] / amount_avg_5d.iloc[i] if amount_avg_5d.iloc[i] > 0 else 1
            volume_trend_5d = data['volume'].iloc[i] / volume_avg_5d.iloc[i] if volume_avg_5d.iloc[i] > 0 else 1
            price_momentum = close_t / close_lag5.iloc[i] - 1
            factor3 = (dollar_volume_trend - volume_trend_5d) * price_momentum
        else:
            factor3 = 0
        
        # Factor 4: Gap-Momentum Reversal Detection
        overnight_gap = data['open'].iloc[i] / close_t1 - 1
        intraday_momentum_current = intraday_momentum.iloc[i]
        volume_spike = data['volume'].iloc[i] / volume_avg_20d.iloc[i] if volume_avg_20d.iloc[i] > 0 else 1
        factor4 = overnight_gap * intraday_momentum_current * volume_spike
        
        # Factor 5: Multi-Timeframe Divergence Signal
        if i >= 20:
            short_momentum = close_t / close_lag5.iloc[i] - 1
            long_momentum = close_t / close_lag20.iloc[i] - 1
            volume_divergence = (data['volume'].iloc[i] / volume_avg_5d.iloc[i] if volume_avg_5d.iloc[i] > 0 else 1) - \
                              (data['volume'].iloc[i] / volume_avg_20d.iloc[i] if volume_avg_20d.iloc[i] > 0 else 1)
            factor5 = (short_momentum - long_momentum) * volume_divergence
        else:
            factor5 = 0
        
        # Combine factors (equal weighting for simplicity)
        combined_factor = (factor1 + factor2 + factor3 + factor4 + factor5) / 5
        factor_values.iloc[i] = combined_factor
    
    return factor_values
