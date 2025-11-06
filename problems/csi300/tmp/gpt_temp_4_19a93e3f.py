import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Adjusted Intraday Divergence with Liquidity Shock factor
    """
    data = df.copy()
    
    # 1. Calculate Intraday Momentum Divergence
    # Short-Term Intraday Momentum
    data['intraday_momentum'] = (data['close'] - data['open']) / data['open']
    
    # Medium-Term Price Momentum (20-day return)
    data['medium_term_momentum'] = data['close'].pct_change(periods=20)
    
    # Momentum Divergence
    data['momentum_divergence'] = data['intraday_momentum'] - data['medium_term_momentum']
    
    # 2. Volatility Adjustment with Regime Detection
    # Rolling Volatility (20-day standard deviation of returns)
    data['daily_returns'] = data['close'].pct_change()
    data['volatility_20d'] = data['daily_returns'].rolling(window=20, min_periods=10).std()
    
    # Volatility Regime Shift Detection
    data['high_low_range'] = data['high'] - data['low']
    
    # Recent volatility (t-8 to t-1)
    recent_vol = data['high_low_range'].shift(1).rolling(window=8, min_periods=4).std()
    
    # Previous volatility (t-15 to t-8)
    previous_vol = data['high_low_range'].shift(8).rolling(window=7, min_periods=4).std()
    
    # Volatility clustering metric
    data['vol_regime_shift'] = recent_vol / previous_vol
    data['vol_regime_shift'] = data['vol_regime_shift'].replace([np.inf, -np.inf], np.nan).fillna(1.0)
    
    # Scale Divergence by Volatility Regime
    data['vol_adjusted_divergence'] = data['momentum_divergence'] / data['volatility_20d']
    data['vol_adjusted_divergence'] = data['vol_adjusted_divergence'] * data['vol_regime_shift']
    
    # 3. Liquidity Shock Confirmation
    # Volume-to-amount ratio
    data['volume_amount_ratio'] = data['volume'] / data['amount']
    
    # Abnormal volume-to-amount ratio (current vs 5-day median)
    data['volume_amount_median_5d'] = data['volume_amount_ratio'].shift(1).rolling(window=5, min_periods=3).median()
    data['liquidity_shock'] = (data['volume_amount_ratio'].shift(1) / data['volume_amount_median_5d']).replace([np.inf, -np.inf], 1.0).fillna(1.0)
    
    # Apply Liquidity Shock Weighting
    data['liquidity_weighted_signal'] = data['vol_adjusted_divergence'] * data['liquidity_shock']
    
    # 4. Volume Profile Integration
    # Volume concentration in upper price ranges (70% of daily range)
    def calculate_upper_volume_ratio(window_data):
        if len(window_data) < 5:
            return np.nan
        total_volume = window_data['volume'].sum()
        if total_volume == 0:
            return 0.0
        
        # Calculate price position for each day in the window
        price_position = (window_data['close'] - window_data['low']) / (window_data['high'] - window_data['low'])
        price_position = price_position.replace([np.inf, -np.inf], 0.5).fillna(0.5)
        
        # Sum volume where price position > 0.7 (upper range)
        upper_volume = window_data.loc[price_position > 0.7, 'volume'].sum()
        return upper_volume / total_volume
    
    # Calculate rolling volume asymmetry (10-day window)
    volume_asymmetry = []
    for i in range(len(data)):
        if i < 10:
            volume_asymmetry.append(np.nan)
            continue
        
        window_start = max(0, i - 10)
        window_data = data.iloc[window_start:i]
        ratio = calculate_upper_volume_ratio(window_data)
        volume_asymmetry.append(ratio)
    
    data['volume_asymmetry'] = volume_asymmetry
    data['volume_asymmetry'] = data['volume_asymmetry'].fillna(0.5)
    
    # Final Signal Construction
    data['final_factor'] = data['liquidity_weighted_signal'] * data['volume_asymmetry']
    
    # Clean and return the factor series
    factor = data['final_factor'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return factor
