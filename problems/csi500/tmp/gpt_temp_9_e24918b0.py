import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multi-timeframe momentum-volume integration
    with volatility-regime adaptive weighting, volume outlier confirmation,
    and price level context integration.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize the factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    # Multi-Timeframe Momentum-Volume Integration
    # Calculate price and volume momentum for different timeframes
    data['price_momentum_3d'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['volume_momentum_3d'] = (data['volume'] - data['volume'].shift(3)) / data['volume'].shift(3)
    
    data['price_momentum_10d'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    data['volume_momentum_10d'] = (data['volume'] - data['volume'].shift(10)) / data['volume'].shift(10)
    
    data['price_momentum_20d'] = (data['close'] - data['close'].shift(20)) / data['close'].shift(20)
    data['volume_momentum_20d'] = (data['volume'] - data['volume'].shift(20)) / data['volume'].shift(20)
    
    # Calculate divergence scores for each timeframe
    def calculate_divergence(price_momentum, volume_momentum):
        if price_momentum > 0 and volume_momentum < 0:
            return 1.0  # Positive divergence
        elif price_momentum < 0 and volume_momentum > 0:
            return -1.0  # Negative divergence
        else:
            return 0.0  # Neutral
    
    data['divergence_3d'] = data.apply(
        lambda x: calculate_divergence(x['price_momentum_3d'], x['volume_momentum_3d']), axis=1
    )
    data['divergence_10d'] = data.apply(
        lambda x: calculate_divergence(x['price_momentum_10d'], x['volume_momentum_10d']), axis=1
    )
    data['divergence_20d'] = data.apply(
        lambda x: calculate_divergence(x['price_momentum_20d'], x['volume_momentum_20d']), axis=1
    )
    
    # Weighted divergence composite
    data['momentum_volume_composite'] = (
        0.5 * data['divergence_3d'] + 
        0.3 * data['divergence_10d'] + 
        0.2 * data['divergence_20d']
    )
    
    # Volatility-Regime Adaptive Weighting
    # Calculate daily returns and 20-day rolling volatility
    data['daily_return'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['volatility_20d'] = data['daily_return'].rolling(window=20).std()
    
    # Calculate 60-day volatility percentiles for regime classification
    data['volatility_percentile'] = data['volatility_20d'].rolling(window=60).apply(
        lambda x: (x.iloc[-1] > x.quantile(0.7)) * 2 + (x.iloc[-1] < x.quantile(0.3)) * 0, 
        raw=False
    )
    
    # Dynamic signal adjustment based on volatility regime
    def adjust_for_volatility(composite_signal, volatility_percentile):
        if volatility_percentile == 2:  # High volatility
            return composite_signal * 0.5  # Reduce momentum signals
        elif volatility_percentile == 0:  # Low volatility
            return composite_signal * 1.3  # Amplify momentum signals
        else:  # Normal volatility
            return composite_signal  # Maintain original
    
    data['volatility_adjusted_signal'] = data.apply(
        lambda x: adjust_for_volatility(x['momentum_volume_composite'], x['volatility_percentile']), 
        axis=1
    )
    
    # Volume Outlier Confirmation System
    data['volume_avg_20d'] = data['volume'].rolling(window=20).mean()
    data['volume_std_20d'] = data['volume'].rolling(window=20).std()
    
    def get_volume_multiplier(volume, volume_avg, volume_std, momentum_signal):
        if volume > (volume_avg + 2 * volume_std):
            volume_category = 'extreme_high'
        elif volume > (volume_avg + volume_std):
            volume_category = 'high'
        elif volume < (volume_avg - volume_std):
            volume_category = 'low'
        else:
            volume_category = 'normal'
        
        strong_momentum = abs(momentum_signal) > 0.5
        
        if strong_momentum and volume_category == 'extreme_high':
            return 2.0
        elif strong_momentum and volume_category == 'high':
            return 1.5
        elif strong_momentum and volume_category == 'normal':
            return 1.0
        elif not strong_momentum and volume_category == 'extreme_high':
            return 0.5
        else:
            return 0.8
    
    data['volume_multiplier'] = data.apply(
        lambda x: get_volume_multiplier(
            x['volume'], x['volume_avg_20d'], x['volume_std_20d'], x['volatility_adjusted_signal']
        ), 
        axis=1
    )
    
    data['volume_confirmed_signal'] = data['volatility_adjusted_signal'] * data['volume_multiplier']
    
    # Price Level Context Integration
    data['high_20d'] = data['close'].rolling(window=20).max()
    data['low_20d'] = data['close'].rolling(window=20).min()
    data['price_position'] = (data['close'] - data['low_20d']) / (data['high_20d'] - data['low_20d'])
    
    def adjust_for_price_position(signal, price_position):
        if price_position > 0.8:  # Near resistance
            if signal > 0:  # Bullish signal
                return signal * 0.6  # Reduce by 40%
            else:  # Bearish signal
                return signal * 1.2  # Amplify by 20%
        elif price_position < 0.2:  # Near support
            if signal < 0:  # Bearish signal
                return signal * 0.6  # Reduce by 40%
            else:  # Bullish signal
                return signal * 1.2  # Amplify by 20%
        else:  # Middle range
            return signal  # No adjustment
    
    data['final_factor'] = data.apply(
        lambda x: adjust_for_price_position(x['volume_confirmed_signal'], x['price_position']), 
        axis=1
    )
    
    # Return the final factor series
    return data['final_factor']
