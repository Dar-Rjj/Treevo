import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns
    data['returns'] = data['close'].pct_change()
    
    # Asymmetric Volatility Response Factor
    # Calculate upside volatility (std of positive returns over 20 days)
    pos_returns = data['returns'].where(data['returns'] > 0)
    upside_vol = pos_returns.rolling(window=20, min_periods=10).std()
    
    # Calculate downside volatility (std of negative returns over 20 days)
    neg_returns = data['returns'].where(data['returns'] < 0)
    downside_vol = neg_returns.rolling(window=20, min_periods=10).std()
    
    # Calculate volume-weighted price trend (10 days)
    vwap = (data['close'] * data['volume']).rolling(window=10).sum() / data['volume'].rolling(window=10).sum()
    price_trend = (data['close'] - vwap) / vwap
    
    # Compute volatility asymmetry ratio with directional confirmation
    vol_asymmetry = upside_vol / downside_vol
    vol_asymmetry_factor = vol_asymmetry * np.sign(price_trend)
    
    # Gap-Fill Momentum Efficiency
    # Calculate gaps (overnight price changes)
    gaps = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Measure gap persistence (days until gap is filled)
    gap_fill_days = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i == 0 or np.isnan(gaps.iloc[i]):
            gap_fill_days.iloc[i] = np.nan
            continue
            
        gap_direction = np.sign(gaps.iloc[i])
        gap_magnitude = abs(gaps.iloc[i])
        
        days_to_fill = 0
        filled = False
        
        for j in range(i, min(i+21, len(data))):
            if j == i:
                continue
                
            current_return = (data['close'].iloc[j] - data['open'].iloc[i]) / data['open'].iloc[i]
            
            # Gap is filled when price crosses the gap
            if gap_direction > 0 and current_return <= 0:
                days_to_fill = j - i
                filled = True
                break
            elif gap_direction < 0 and current_return >= 0:
                days_to_fill = j - i
                filled = True
                break
        
        if not filled:
            days_to_fill = 20  # Maximum lookback
        
        gap_fill_days.iloc[i] = days_to_fill
    
    # Compute gap-fill speed (inverse of days to fill)
    gap_fill_speed = 1 / (gap_fill_days + 1)
    
    # Calculate volume change ratio during fill period
    volume_ratio = data['volume'].rolling(window=5).mean() / data['volume'].rolling(window=20).mean()
    
    # Combine gap factors
    gap_momentum_factor = gap_fill_speed * volume_ratio * abs(gaps)
    
    # Pressure Release Oscillator
    # Calculate compression intensity (narrow range days / total days over 20)
    daily_range = (data['high'] - data['low']) / data['close']
    avg_range = daily_range.rolling(window=20).mean()
    narrow_range_days = (daily_range < avg_range * 0.7).rolling(window=20).sum()
    compression_intensity = narrow_range_days / 20
    
    # Measure release magnitude (range expansion following compression)
    release_magnitude = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i < 20:
            release_magnitude.iloc[i] = np.nan
            continue
            
        # Look for compression periods
        if compression_intensity.iloc[i] > 0.6:  # High compression
            # Find next 5 days maximum range expansion
            max_expansion = 0
            for j in range(i+1, min(i+6, len(data))):
                expansion = daily_range.iloc[j] / avg_range.iloc[i]
                if expansion > max_expansion:
                    max_expansion = expansion
            
            release_magnitude.iloc[i] = max_expansion if max_expansion > 0 else 1
        else:
            release_magnitude.iloc[i] = 1
    
    # Weight by directional bias and volume confirmation
    ma_20 = data['close'].rolling(window=20).mean()
    trend_direction = np.sign(data['close'] - ma_20)
    volume_confirmation = data['volume'] / data['volume'].rolling(window=20).mean()
    
    pressure_release_factor = compression_intensity * release_magnitude * trend_direction * volume_confirmation
    
    # Volume-Weighted Price Elasticity
    # Calculate volume surprise
    volume_surprise = data['volume'] / data['volume'].rolling(window=20).mean()
    
    # Calculate price impact
    price_impact = abs(data['returns'])
    
    # Compute price elasticity
    price_elasticity = price_impact / (volume_surprise + 1e-6)
    
    # Apply trend persistence filter
    returns_5d = data['returns'].rolling(window=5).sum()
    direction_consistency = (np.sign(data['returns'].rolling(window=5).mean()) == np.sign(returns_5d)).astype(float)
    
    elasticity_factor = price_elasticity * direction_consistency
    
    # Combine all factors with equal weighting
    final_factor = (
        vol_asymmetry_factor.fillna(0) + 
        gap_momentum_factor.fillna(0) + 
        pressure_release_factor.fillna(0) + 
        elasticity_factor.fillna(0)
    ) / 4
    
    return final_factor
