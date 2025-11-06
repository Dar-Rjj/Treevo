import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate daily returns for volatility computation
    daily_returns = df['close'].pct_change()
    
    # Calculate True Range
    high_low_range = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    true_range = pd.concat([high_low_range, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Compute Intraday Return and Range Efficiency Ratio
    intraday_return = (df['close'] - df['open']) / df['open']
    range_efficiency = abs(intraday_return) / true_range
    range_efficiency = range_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Calculate multi-timeframe volatility
    short_term_vol = daily_returns.rolling(window=5, min_periods=3).std()
    medium_term_vol = daily_returns.rolling(window=20, min_periods=10).std()
    
    # Compute Volatility Regime Ratio
    volatility_ratio = short_term_vol / medium_term_vol
    
    # Calculate Volume Intensity
    avg_volume_20d = df['volume'].rolling(window=20, min_periods=10).mean()
    volume_intensity = df['volume'] / avg_volume_20d
    
    # Regress range efficiency on volatility regime to extract volatility-independent component
    # Use rolling regression residuals to capture pure efficiency
    window_size = 20
    volatility_independent_efficiency = pd.Series(index=df.index, dtype=float)
    
    for i in range(window_size, len(df)):
        if i >= window_size:
            window_data = range_efficiency.iloc[i-window_size:i]
            vol_ratio_data = volatility_ratio.iloc[i-window_size:i]
            
            # Simple linear regression using covariance approach
            if not (window_data.isna().any() or vol_ratio_data.isna().any()):
                cov_matrix = np.cov(window_data, vol_ratio_data)
                if cov_matrix[1,1] > 0:
                    beta = cov_matrix[0,1] / cov_matrix[1,1]
                    predicted = beta * volatility_ratio.iloc[i]
                    residual = range_efficiency.iloc[i] - predicted
                    volatility_independent_efficiency.iloc[i] = residual
    
    # Apply volume-intensity multiplier
    volume_adjusted_efficiency = volatility_independent_efficiency * volume_intensity
    
    # Generate regime-based signals
    final_alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if pd.notna(volatility_ratio.iloc[i]) and pd.notna(volume_adjusted_efficiency.iloc[i]):
            # High Volatility Compression regime
            if volatility_ratio.iloc[i] < 1.0:
                # Focus on range efficiency with volume confirmation
                if volume_intensity.iloc[i] > 1.0:
                    final_alpha.iloc[i] = volume_adjusted_efficiency.iloc[i] * 1.5
                else:
                    final_alpha.iloc[i] = volume_adjusted_efficiency.iloc[i] * 0.7
            
            # Volatility Expansion regime
            else:
                # Emphasize momentum quality with volume filter
                if volume_intensity.iloc[i] > 1.2:
                    final_alpha.iloc[i] = volume_adjusted_efficiency.iloc[i] * 1.2
                else:
                    final_alpha.iloc[i] = volume_adjusted_efficiency.iloc[i] * 0.5
    
    return final_alpha
