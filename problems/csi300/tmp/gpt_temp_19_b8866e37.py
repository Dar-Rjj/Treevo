import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining amplitude-frequency dynamics, volume-volatility confluence,
    price-volume fractality, momentum asymmetry, and microstructure pressure.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Rolling window parameters
    short_window = 5
    medium_window = 20
    long_window = 60
    
    for i in range(long_window, len(df)):
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # 1. Amplitude-Frequency Price Dynamics
        # Local extrema density - count peaks and troughs in recent period
        highs = current_data['high'].iloc[-short_window:]
        lows = current_data['low'].iloc[-short_window:]
        
        # Simple peak detection (local maxima)
        peaks = 0
        for j in range(1, len(highs)-1):
            if highs.iloc[j] > highs.iloc[j-1] and highs.iloc[j] > highs.iloc[j+1]:
                peaks += 1
        
        # Simple trough detection (local minima)
        troughs = 0
        for j in range(1, len(lows)-1):
            if lows.iloc[j] < lows.iloc[j-1] and lows.iloc[j] < lows.iloc[j+1]:
                troughs += 1
        
        extrema_density = (peaks + troughs) / short_window
        
        # Cycle duration - measure volatility-adjusted price swings
        recent_volatility = current_data['close'].iloc[-medium_window:].pct_change().std()
        if recent_volatility > 0:
            price_range = (highs.max() - lows.min()) / current_data['close'].iloc[-short_window]
            volatility_adjusted_swing = price_range / recent_volatility
        else:
            volatility_adjusted_swing = 0
        
        # 2. Volume-Volatility Confluence
        recent_volume = current_data['volume'].iloc[-medium_window:]
        recent_returns = current_data['close'].iloc[-medium_window:].pct_change().dropna()
        
        # Volume-volatility correlation
        if len(recent_returns) > 1:
            volume_vol_corr = np.corrcoef(recent_volume.iloc[1:], np.abs(recent_returns))[0,1]
        else:
            volume_vol_corr = 0
        
        # Volatility per unit volume
        if recent_volume.mean() > 0:
            vol_per_volume = recent_returns.std() / recent_volume.mean()
        else:
            vol_per_volume = 0
        
        # 3. Price-Volume Fractality
        # Multi-scale patterns using different rolling windows
        short_ma = current_data['close'].iloc[-short_window:].mean()
        medium_ma = current_data['close'].iloc[-medium_window:].mean()
        long_ma = current_data['close'].iloc[-long_window:].mean()
        
        # Fractal dimension proxy - ratio of short to medium term volatility
        short_vol = current_data['close'].iloc[-short_window:].pct_change().std()
        medium_vol = current_data['close'].iloc[-medium_window:].pct_change().std()
        
        if medium_vol > 0:
            fractal_ratio = short_vol / medium_vol
        else:
            fractal_ratio = 0
        
        # 4. Momentum Asymmetry
        recent_closes = current_data['close'].iloc[-medium_window:]
        returns = recent_closes.pct_change().dropna()
        
        # Separate up and down moves
        up_moves = returns[returns > 0]
        down_moves = returns[returns < 0]
        
        # Asymmetric momentum characteristics
        if len(up_moves) > 0 and len(down_moves) > 0:
            momentum_skew = (up_moves.mean() - abs(down_moves.mean())) / (up_moves.std() + abs(down_moves.std()))
        else:
            momentum_skew = 0
        
        # Up vs down momentum persistence
        up_persistence = len(up_moves) / len(returns) if len(returns) > 0 else 0
        
        # 5. Microstructure Pressure
        recent_amount = current_data['amount'].iloc[-short_window:]
        recent_prices = current_data['close'].iloc[-short_window:]
        
        # Price impact proxy - relationship between amount and price changes
        if len(recent_prices) > 1:
            price_changes = recent_prices.pct_change().dropna()
            if len(price_changes) > 0 and recent_amount.iloc[1:].std() > 0:
                price_impact = np.corrcoef(recent_amount.iloc[1:], np.abs(price_changes))[0,1]
            else:
                price_impact = 0
        else:
            price_impact = 0
        
        # Volume-weighted pressure
        if recent_volume.iloc[-short_window:].sum() > 0:
            vwap = (current_data['amount'].iloc[-short_window:].sum() / 
                   current_data['volume'].iloc[-short_window:].sum())
            pressure = (current_data['close'].iloc[-1] - vwap) / vwap
        else:
            pressure = 0
        
        # Combine all components into final factor
        factor_value = (
            extrema_density * 0.15 +
            volatility_adjusted_swing * 0.12 +
            volume_vol_corr * 0.18 +
            vol_per_volume * 0.10 +
            fractal_ratio * 0.12 +
            momentum_skew * 0.15 +
            price_impact * 0.10 +
            pressure * 0.08
        )
        
        result.iloc[i] = factor_value
    
    # Fill initial values with NaN
    result.iloc[:long_window] = np.nan
    
    return result
