import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Volatility Asymmetry & Compression
    # Upside Volatility: avg(max(0, High_t - Close_t)) over 10 days
    upside_vol = (data['high'] - data['close']).clip(lower=0).rolling(window=10, min_periods=5).mean()
    
    # Downside Volatility: avg(max(0, Close_t - Low_t)) over 10 days
    downside_vol = (data['close'] - data['low']).clip(lower=0).rolling(window=10, min_periods=5).mean()
    
    # Volatility Asymmetry: Upside Volatility / Downside Volatility
    # Avoid division by zero
    volatility_asymmetry = upside_vol / downside_vol.replace(0, np.nan)
    
    # Volatility Compression: 5-day volatility / 15-day volatility
    vol_5d = data['close'].pct_change().rolling(window=5, min_periods=3).std()
    vol_15d = data['close'].pct_change().rolling(window=15, min_periods=8).std()
    volatility_compression = vol_5d / vol_15d.replace(0, np.nan)
    
    # Base Factor: Volatility Asymmetry × Volatility Compression
    base_factor = volatility_asymmetry * volatility_compression
    
    # Breakout Efficiency
    # Intraday Efficiency: (Close_t - Low_t) / (High_t - Low_t)
    intraday_efficiency = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volume Confirmation: Volume_t / 5-day average volume
    volume_confirmation = data['volume'] / data['volume'].rolling(window=5, min_periods=3).mean()
    
    # Efficiency Signal: Intraday Efficiency × Volume Confirmation
    efficiency_signal = intraday_efficiency * volume_confirmation
    
    # With Breakout: Base Factor × Efficiency Signal
    with_breakout = base_factor * efficiency_signal
    
    # Momentum Quality Assessment
    # Calculate daily returns
    returns = data['close'].pct_change()
    
    # Directional Persistence: consecutive same-sign returns over 5 days
    def count_consecutive_same_sign(series, window=5):
        signs = np.sign(series)
        result = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i < window - 1:
                result.iloc[i] = np.nan
            else:
                window_signs = signs.iloc[i-window+1:i+1]
                # Count consecutive same signs from the end
                current_sign = window_signs.iloc[-1]
                count = 0
                for j in range(len(window_signs)-1, -1, -1):
                    if window_signs.iloc[j] == current_sign and current_sign != 0:
                        count += 1
                    else:
                        break
                result.iloc[i] = count
        return result
    
    directional_persistence = count_consecutive_same_sign(returns, window=5)
    
    # Return-to-Volatility: 5-day return / 5-day volatility
    return_5d = data['close'].pct_change(periods=5)
    vol_5d_returns = returns.rolling(window=5, min_periods=3).std()
    return_to_volatility = return_5d / vol_5d_returns.replace(0, np.nan)
    
    # Quality Signal: Directional Persistence × Return-to-Volatility
    quality_signal = directional_persistence * return_to_volatility
    
    # With Momentum: With Breakout × Quality Signal
    with_momentum = with_breakout * quality_signal
    
    # Microstructure Pressure
    # Price Rejection: (High_t - Close_t) / (Close_t - Low_t)
    price_rejection = (data['high'] - data['close']) / (data['close'] - data['low']).replace(0, np.nan)
    
    # Volume Concentration: (High_t - Low_t) / Volume_t
    volume_concentration = (data['high'] - data['low']) / data['volume'].replace(0, np.nan)
    
    # Pressure Score: Price Rejection × Volume Concentration
    pressure_score = price_rejection * volume_concentration
    
    # Final Alpha: With Momentum / Pressure Score
    final_alpha = with_momentum / pressure_score.replace(0, np.nan)
    
    return final_alpha
