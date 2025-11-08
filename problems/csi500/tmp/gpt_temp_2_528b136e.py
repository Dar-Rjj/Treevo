import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Identification
    # Calculate daily returns
    daily_returns = data['close'] / data['close'].shift(1) - 1
    
    # Calculate 10-day realized volatility
    realized_vol_10 = daily_returns.rolling(window=10, min_periods=5).std()
    
    # Calculate 20-day median of 10-day realized volatility
    vol_median = realized_vol_10.rolling(window=20, min_periods=10).median()
    
    # Classify volatility regime
    high_vol_regime = realized_vol_10 > vol_median
    
    # Momentum Breakout Detection
    # Multi-timeframe momentum analysis
    short_term_return = data['close'] / data['close'].shift(5) - 1
    medium_term_return = data['close'] / data['close'].shift(20) - 1
    long_term_return = data['close'] / data['close'].shift(60) - 1
    
    # Breakout pattern identification
    momentum_divergence = medium_term_return > short_term_return
    trend_confirmation = long_term_return > medium_term_return
    
    # Volatility expansion
    realized_vol_30 = daily_returns.rolling(window=30, min_periods=15).std()
    vol_expansion = (realized_vol_10 / realized_vol_30) > 1.2
    
    # Price-Volume Convergence Components
    # Volume-Price Alignment
    daily_price_change = data['close'] - data['close'].shift(1)
    signed_volume = data['volume'] * np.sign(daily_price_change)
    volume_price_alignment = signed_volume.rolling(window=3, min_periods=2).sum()
    
    # Intraday Price Efficiency
    intraday_price_efficiency = (data['close'] - data['open']) / (data['high'] - data['low'])
    intraday_price_efficiency = intraday_price_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Volume Momentum Analysis
    volume_acceleration = ((data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1)) - \
                         ((data['volume'].shift(1) - data['volume'].shift(2)) / data['volume'].shift(2))
    
    volume_surge = data['volume'] / data['volume'].shift(5) - 1
    
    # Volume surge detection using 5-day SMA
    volume_sma_5 = data['volume'].rolling(window=5, min_periods=3).mean()
    volume_surge_detected = data['volume'] > (1.2 * volume_sma_5)
    
    # Price efficiency for low volatility regime
    price_efficiency_5day = abs(data['close'] - data['open']).rolling(window=5, min_periods=3).mean() / \
                           (data['high'] - data['low']).rolling(window=5, min_periods=3).mean()
    
    # Regime-Adaptive Factor Combination
    alpha_factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if pd.isna(realized_vol_10.iloc[i]) or pd.isna(vol_median.iloc[i]):
            alpha_factor.iloc[i] = np.nan
            continue
            
        if high_vol_regime.iloc[i]:
            # High Volatility Regime
            momentum_component = momentum_divergence.iloc[i] * volume_price_alignment.iloc[i]
            
            # Apply volatility amplification
            if not pd.isna(realized_vol_10.iloc[i]) and not pd.isna(realized_vol_30.iloc[i]) and realized_vol_30.iloc[i] != 0:
                vol_amplification = realized_vol_10.iloc[i] / realized_vol_30.iloc[i]
                momentum_component *= vol_amplification
            
            # Volume surge multiplier
            volume_multiplier = 1.5 if volume_surge_detected.iloc[i] else 1.0
            
            base_factor = momentum_component * volume_multiplier
            
        else:
            # Low Volatility Regime
            momentum_component = momentum_divergence.iloc[i] * intraday_price_efficiency.iloc[i]
            
            # Apply efficiency weighting
            if not pd.isna(price_efficiency_5day.iloc[i]):
                efficiency_weight = price_efficiency_5day.iloc[i]
                momentum_component *= efficiency_weight
            
            base_factor = momentum_component
        
        # Conditional Multipliers
        final_factor = base_factor
        
        # Volume Acceleration Multiplier
        if not pd.isna(volume_acceleration.iloc[i]) and volume_acceleration.iloc[i] > 0:
            final_factor *= (1 + volume_acceleration.iloc[i])
        
        # Trend Confirmation Multiplier
        if not pd.isna(long_term_return.iloc[i]):
            final_factor *= (1 + long_term_return.iloc[i])
        
        # Intraday Momentum Multiplier
        if not pd.isna(intraday_price_efficiency.iloc[i]) and intraday_price_efficiency.iloc[i] > 0.5:
            final_factor *= (1 + intraday_price_efficiency.iloc[i])
        
        alpha_factor.iloc[i] = final_factor
    
    return alpha_factor
