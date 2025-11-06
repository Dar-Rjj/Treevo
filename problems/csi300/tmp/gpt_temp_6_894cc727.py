import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize result series
    result = pd.Series(index=data.index, dtype=float)
    
    # Calculate returns for volatility calculations
    data['ret'] = data['close'] / data['close'].shift(1) - 1
    
    # Calculate rolling statistics
    data['volume_median_20'] = data['volume'].rolling(window=20, min_periods=10).median()
    data['volume_avg_4'] = data['volume'].rolling(window=4, min_periods=2).mean()
    data['volume_avg_9'] = data['volume'].rolling(window=9, min_periods=5).mean()
    data['volume_std_9'] = data['volume'].rolling(window=9, min_periods=5).std()
    data['volatility_short'] = data['ret'].rolling(window=5, min_periods=3).std()
    data['volatility_medium'] = data['ret'].rolling(window=10, min_periods=5).std()
    
    # Calculate rolling high/low for range breakout
    data['high_4'] = data['high'].rolling(window=4, min_periods=2).max()
    data['low_4'] = data['low'].rolling(window=4, min_periods=2).min()
    
    # Calculate up-volume dominance
    def calc_up_volume_dominance(window):
        if len(window) < 2:
            return np.nan
        up_volume = 0
        total_volume = 0
        for i in range(1, len(window)):
            if window['close'].iloc[i] > window['close'].iloc[i-1]:
                up_volume += window['volume'].iloc[i]
            total_volume += window['volume'].iloc[i]
        return up_volume / total_volume if total_volume > 0 else 0
    
    # Calculate all factors for each day
    for i in range(len(data)):
        if i < 10:  # Need sufficient history
            result.iloc[i] = 0
            continue
            
        current = data.iloc[i]
        prev = data.iloc[i-1] if i > 0 else None
        
        # Factor 1: Liquidity Shock with Price Resilience
        if not pd.isna(current['volume_median_20']) and current['volume_median_20'] > 0:
            abnormal_volume = current['volume'] / current['volume_median_20'] - 1
            price_impact_efficiency = abs(current['close'] / prev['close'] - 1) / (current['amount'] / current['volume']) if current['amount'] > 0 else 0
            liquidity_shock = abnormal_volume * price_impact_efficiency
            
            intraday_recovery = (current['close'] - current['low']) / (current['high'] - current['low']) if current['high'] != current['low'] else 0
            overnight_gap_persistence = (current['open'] / prev['close'] - 1) * (current['close'] / current['open'] - 1)
            price_recovery = intraday_recovery * overnight_gap_persistence
            
            factor1 = liquidity_shock * price_recovery
        else:
            factor1 = 0
        
        # Factor 2: Volatility Regime Shift with Volume Confirmation
        if not pd.isna(current['volatility_short']) and not pd.isna(current['volatility_medium']) and current['volatility_medium'] > 0:
            volatility_ratio = current['volatility_short'] / current['volatility_medium']
            
            volume_trend = current['volume'] / current['volume_avg_4'] - 1 if not pd.isna(current['volume_avg_4']) and current['volume_avg_4'] > 0 else 0
            volume_volatility = current['volume_std_9'] / current['volume_avg_9'] if not pd.isna(current['volume_std_9']) and not pd.isna(current['volume_avg_9']) and current['volume_avg_9'] > 0 else 0
            volume_pattern = volume_trend * volume_volatility
            
            factor2 = volatility_ratio * volume_pattern
        else:
            factor2 = 0
        
        # Factor 3: Price-Momentum Divergence with Trade Intensity
        if i >= 5:
            short_momentum = current['close'] / data.iloc[i-2]['close'] - 1
            medium_momentum = current['close'] / data.iloc[i-5]['close'] - 1
            momentum_divergence = short_momentum - medium_momentum
            
            large_trade_ratio = current['amount'] / (current['volume'] * current['close']) if current['volume'] > 0 and current['close'] > 0 else 0
            trade_concentration = (current['high'] - current['low']) / abs(current['close'] / prev['close'] - 1) if abs(current['close'] / prev['close'] - 1) > 0 else 0
            trade_intensity = large_trade_ratio * trade_concentration
            
            factor3 = momentum_divergence * trade_intensity
        else:
            factor3 = 0
        
        # Factor 4: Range Breakout Efficiency with Volume Asymmetry
        if not pd.isna(current['high_4']) and not pd.isna(current['low_4']) and current['high_4'] != current['low_4']:
            price_position = (current['close'] - current['low_4']) / (current['high_4'] - current['low_4'])
            breakout_strength = (current['close'] / current['high_4'] - 1) * (current['close'] / current['low_4'] - 1)
            range_breakout = price_position * breakout_strength
            
            # Calculate up-volume dominance for last 4 days
            window_data = data.iloc[max(0, i-3):i+1]
            up_volume_dominance = calc_up_volume_dominance(window_data)
            
            volume_momentum = current['volume'] / data.iloc[i-9:i-4]['volume'].mean() - 1 if i >= 9 else 0
            volume_asymmetry = up_volume_dominance * volume_momentum
            
            factor4 = range_breakout * volume_asymmetry
        else:
            factor4 = 0
        
        # Combine factors (equal weighting)
        result.iloc[i] = 0.25 * factor1 + 0.25 * factor2 + 0.25 * factor3 + 0.25 * factor4
    
    # Fill initial NaN values with 0
    result = result.fillna(0)
    
    return result
