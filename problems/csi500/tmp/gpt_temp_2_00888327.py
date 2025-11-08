import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility Regime Asymmetry
    # Calculate Upside Volatility Component (10-day window)
    upside_vol = pd.Series(index=data.index, dtype=float)
    downside_vol = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 9:  # Need at least 10 days of data
            window = data.iloc[i-9:i+1]
            upside_values = [max(0, row['high'] - row['close']) for _, row in window.iterrows()]
            downside_values = [max(0, row['close'] - row['low']) for _, row in window.iterrows()]
            upside_vol.iloc[i] = np.mean(upside_values)
            downside_vol.iloc[i] = np.mean(downside_values)
    
    # Generate Volatility Asymmetry Ratio
    volatility_asymmetry = upside_vol / downside_vol
    volatility_asymmetry = volatility_asymmetry.replace([np.inf, -np.inf], np.nan)
    
    # 2. Price-Volume Efficiency Patterns
    # Intraday Efficiency Metric
    intraday_efficiency = (data['close'] - data['low']) / (data['high'] - data['low'])
    intraday_efficiency = intraday_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Volume Distribution Efficiency (5-day percentile within 20-day window)
    volume_percentile = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 19:  # Need at least 20 days of data
            volume_window = data['volume'].iloc[i-19:i+1]
            current_volume = data['volume'].iloc[i]
            volume_percentile.iloc[i] = (volume_window <= current_volume).sum() / len(volume_window)
    
    # Efficiency Divergence Signal
    efficiency_divergence = intraday_efficiency * volume_percentile
    
    # 3. Momentum Persistence Characteristics
    # Directional Persistence Score
    directional_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:  # Need at least 5 days of data
            window = data['close'].iloc[i-4:i+1]
            returns = window.pct_change().dropna()
            if len(returns) >= 2:
                consecutive_count = 1
                for j in range(1, len(returns)):
                    if returns.iloc[j] * returns.iloc[j-1] > 0:  # Same direction
                        consecutive_count += 1
                    else:
                        break
                directional_persistence.iloc[i] = consecutive_count
    
    # Momentum Quality (5-day return-to-volatility ratio)
    momentum_quality = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:  # Need at least 5 days of data
            window = data['close'].iloc[i-4:i+1]
            returns = window.pct_change().dropna()
            if len(returns) >= 1:
                total_return = (window.iloc[-1] / window.iloc[0]) - 1
                volatility = returns.std()
                if volatility > 0:
                    momentum_quality.iloc[i] = total_return / volatility
    
    # Persistence Quality Signal
    persistence_quality = directional_persistence * momentum_quality
    
    # 4. Market Microstructure Pressure
    # Price Rejection Signals
    price_rejection = (data['high'] - data['close']) / (data['close'] - data['low'])
    price_rejection = price_rejection.replace([np.inf, -np.inf], np.nan)
    
    # Volume Concentration Timing
    volume_concentration = (data['high'] - data['low']) / data['volume']
    volume_concentration = volume_concentration.replace([np.inf, -np.inf], np.nan)
    
    # Microstructure Pressure Score
    microstructure_pressure = price_rejection * volume_concentration
    
    # 5. Generate Composite Alpha Factor
    # Combine Volatility Regime with Efficiency
    vol_efficiency_component = volatility_asymmetry * efficiency_divergence
    
    # Apply Momentum Quality Filter
    momentum_filtered = vol_efficiency_component * persistence_quality
    
    # Incorporate Microstructure Adjustment
    alpha_factor = momentum_filtered / microstructure_pressure
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    
    return alpha_factor
