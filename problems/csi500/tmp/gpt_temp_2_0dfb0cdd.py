import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Regime-Transition Acceleration Divergence factor
    Captures regime transition timing advantages through multi-dimensional acceleration analysis
    """
    
    # Price Acceleration Hierarchy
    def calc_price_acceleration(close, short_period, long_period):
        short_ret = close / close.shift(short_period)
        long_ret = close.shift(short_period) / close.shift(long_period)
        return short_ret - long_ret
    
    # Volume/Amount Acceleration Hierarchy
    def calc_volume_amount_acceleration(data, short_period, long_period):
        short_ratio = data / data.shift(short_period)
        long_ratio = data.shift(short_period) / data.shift(long_period)
        return short_ratio - long_ratio
    
    close = df['close']
    volume = df['volume']
    amount = df['amount']
    high = df['high']
    low = df['low']
    
    # Multi-Timeframe Acceleration Framework
    price_accel = {}
    volume_accel = {}
    amount_accel = {}
    
    # Define timeframe pairs
    timeframes = [
        ('ultra_short', 3, 6),
        ('short', 5, 10),
        ('medium', 10, 20),
        ('long', 20, 40)
    ]
    
    for name, short_p, long_p in timeframes:
        price_accel[name] = calc_price_acceleration(close, short_p, long_p)
        volume_accel[name] = calc_volume_amount_acceleration(volume, short_p, long_p)
        amount_accel[name] = calc_volume_amount_acceleration(amount, short_p, long_p)
    
    # Regime Transition Detection System
    def calc_transition_magnitude(accel_dict):
        """Calculate transition magnitude as sum of absolute acceleration changes"""
        magnitudes = {}
        for timeframe in ['ultra_short', 'short', 'medium', 'long']:
            magnitudes[timeframe] = accel_dict[timeframe].diff().abs()
        return sum(magnitudes.values())
    
    price_transition = calc_transition_magnitude(price_accel)
    volume_transition = calc_transition_magnitude(volume_accel)
    amount_transition = calc_transition_magnitude(amount_accel)
    
    # Range volatility transition
    range_vol = (high - low) / close
    range_transition = (range_vol - range_vol.shift(5)).abs()
    
    # Composite transition score
    composite_transition = (
        price_transition.rolling(5).mean() * 0.4 +
        volume_transition.rolling(5).mean() * 0.3 +
        amount_transition.rolling(5).mean() * 0.2 +
        range_transition.rolling(5).mean() * 0.1
    )
    
    # Transition Type Classification
    def classify_transition_type(price_acc, volume_acc):
        """Classify transition types based on price and volume acceleration directions"""
        price_dir = np.sign(price_acc)
        volume_dir = np.sign(volume_acc)
        
        bullish = (price_dir > 0) & (volume_dir < 0)
        bearish = (price_dir < 0) & (volume_dir > 0)
        confirmation = (price_dir == volume_dir) & (price_dir != 0)
        divergence = (price_dir != volume_dir) & (price_dir != 0) & (volume_dir != 0)
        
        return bullish.astype(int) - bearish.astype(int) + confirmation.astype(int) * 0.5
    
    transition_type = classify_transition_type(
        price_accel['medium'], volume_accel['medium']
    )
    
    # Dynamic Cross-Sectional Comparison
    def calc_acceleration_divergence(price_acc, volume_acc):
        """Calculate price-volume acceleration divergence"""
        return price_acc - volume_acc
    
    # Multi-timeframe divergence
    divergences = {}
    for timeframe in timeframes:
        name = timeframe[0]
        divergences[name] = calc_acceleration_divergence(
            price_accel[name], volume_accel[name]
        )
    
    # Cross-sectional ranking within rolling window
    def cross_sectional_rank(data, window=20):
        """Calculate cross-sectional percentile rank"""
        return data.rolling(window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
    
    # Acceleration persistence scoring
    def calc_persistence_score(acceleration_series, window=10):
        """Calculate consistency of acceleration direction"""
        direction = np.sign(acceleration_series)
        return direction.rolling(window).apply(
            lambda x: (x == x.iloc[-1]).mean(), raw=False
        )
    
    # Adaptive Signal Construction
    def dynamic_timeframe_weighting(transition_score):
        """Dynamic weighting based on transition periods"""
        # High transition periods emphasize short-term timeframes
        transition_level = pd.cut(transition_score, 
                                bins=[-np.inf, transition_score.quantile(0.3), 
                                      transition_score.quantile(0.7), np.inf],
                                labels=['low', 'medium', 'high'])
        
        weights = {}
        for timeframe in ['ultra_short', 'short', 'medium', 'long']:
            if timeframe == 'ultra_short':
                weights[timeframe] = np.where(transition_level == 'high', 0.4, 
                                            np.where(transition_level == 'medium', 0.3, 0.2))
            elif timeframe == 'short':
                weights[timeframe] = np.where(transition_level == 'high', 0.3, 
                                            np.where(transition_level == 'medium', 0.35, 0.3))
            elif timeframe == 'medium':
                weights[timeframe] = np.where(transition_level == 'high', 0.2, 
                                            np.where(transition_level == 'medium', 0.25, 0.3))
            else:  # long
                weights[timeframe] = np.where(transition_level == 'high', 0.1, 
                                            np.where(transition_level == 'medium', 0.1, 0.2))
        
        return weights
    
    # Calculate dynamic weights
    weights = dynamic_timeframe_weighting(composite_transition)
    
    # Multi-timeframe acceleration synthesis
    weighted_divergence = sum(
        divergences[timeframe] * weights[timeframe] 
        for timeframe in ['ultra_short', 'short', 'medium', 'long']
    )
    
    # Regime-transition signal enhancement
    transition_confidence = composite_transition.rolling(10).mean()
    enhanced_signal = weighted_divergence * (1 + transition_confidence)
    
    # Cross-sectional signal refinement
    cross_sectional_ranked = cross_sectional_rank(enhanced_signal, 20)
    persistence_score = calc_persistence_score(weighted_divergence, 10)
    
    # Final factor construction
    factor = (
        enhanced_signal * 0.5 +
        cross_sectional_ranked * 0.3 +
        persistence_score * 0.2
    ) * transition_type.abs()
    
    # Normalize the factor
    factor = (factor - factor.rolling(60).mean()) / factor.rolling(60).std()
    
    return factor
