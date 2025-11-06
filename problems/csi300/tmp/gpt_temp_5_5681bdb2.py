import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Shift Adaptive Momentum Factor
    Combines volatility regime detection, trend analysis, and volume-price asymmetry
    to generate adaptive momentum signals.
    """
    df = data.copy()
    
    # Multi-Timeframe Regime Detection
    # Volatility Regime Classification
    df['high_low_range'] = df['high'] - df['low']
    df['short_term_vol'] = df['high_low_range'].rolling(window=5).mean()
    
    df['returns'] = df['close'].pct_change()
    df['medium_term_vol'] = df['returns'].rolling(window=20).std()
    
    # Regime Classification Logic
    conditions = [
        df['short_term_vol'] > df['medium_term_vol'],
        df['short_term_vol'] < (0.5 * df['medium_term_vol'])
    ]
    choices = ['high', 'low']
    df['volatility_regime'] = np.select(conditions, choices, default='normal')
    
    # Trend Regime Classification
    def calculate_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series) - window + 1):
            y = series.iloc[i:i+window].values
            if len(y) == window:
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=series.index[window-1:])
    
    df['short_trend'] = calculate_slope(df['close'], 3)
    df['medium_trend'] = calculate_slope(df['close'], 10)
    df['long_trend'] = calculate_slope(df['close'], 20)
    
    # Trend Regime Logic
    trend_conditions = [
        (df['short_trend'] * df['medium_trend'] > 0) & 
        (df['medium_trend'] * df['long_trend'] > 0) & 
        (df['short_trend'] * df['long_trend'] > 0),
        (abs(df['short_trend']) < 0.001) & 
        (abs(df['medium_trend']) < 0.001) & 
        (abs(df['long_trend']) < 0.001)
    ]
    trend_choices = ['strong', 'no_trend']
    df['trend_regime'] = np.select(trend_conditions, trend_choices, default='weak')
    
    # Volume-Price Asymmetry Analysis
    # Upside Volume Pressure
    up_days = df['close'] > df['close'].shift(1)
    df['upside_pressure'] = np.where(up_days, df['returns'] * df['volume'], 0)
    df['avg_upside_pressure'] = df['upside_pressure'].rolling(window=5).mean()
    
    # Downside Volume Pressure
    down_days = df['close'] < df['close'].shift(1)
    df['downside_pressure'] = np.where(down_days, abs(df['returns']) * df['volume'], 0)
    df['avg_downside_pressure'] = df['downside_pressure'].rolling(window=5).mean()
    
    # Volume Asymmetry Ratio
    df['volume_asymmetry'] = df['avg_upside_pressure'] / (df['avg_downside_pressure'] + 1e-8)
    df['asymmetry_momentum'] = df['volume_asymmetry'] / df['volume_asymmetry'].rolling(window=5).mean()
    
    # Price-Volume Divergence
    price_trend_direction = np.sign(df['medium_trend'])
    volume_asymmetry_direction = np.sign(df['volume_asymmetry'] - 1)
    df['price_volume_divergence'] = np.where(
        price_trend_direction == volume_asymmetry_direction,
        abs(df['medium_trend']) * df['asymmetry_momentum'],
        -abs(df['medium_trend']) * df['asymmetry_momentum']
    )
    
    # Regime-Adaptive Signal Generation
    # Volatility Regime Weighting
    volatility_weights = {
        'high': {'volume_weight': 0.7, 'trend_weight': 0.3, 'momentum_window': 5},
        'low': {'volume_weight': 0.3, 'trend_weight': 0.7, 'momentum_window': 20},
        'normal': {'volume_weight': 0.5, 'trend_weight': 0.5, 'momentum_window': 10}
    }
    
    # Trend Regime Adaptation
    trend_multipliers = {
        'strong': 1.2,
        'weak': 1.0,
        'no_trend': 0.8
    }
    
    # Calculate regime-adaptive components
    df['volatility_component'] = 0.0
    df['trend_component'] = 0.0
    df['volume_component'] = 0.0
    
    for regime in ['high', 'low', 'normal']:
        mask = df['volatility_regime'] == regime
        if mask.any():
            window = volatility_weights[regime]['momentum_window']
            vol_weight = volatility_weights[regime]['volume_weight']
            trend_weight = volatility_weights[regime]['trend_weight']
            
            df.loc[mask, 'volatility_component'] = df.loc[mask, 'returns'].rolling(window=window).mean()
            df.loc[mask, 'trend_component'] = df.loc[mask, 'medium_trend']
            df.loc[mask, 'volume_component'] = df.loc[mask, 'price_volume_divergence'] * vol_weight
    
    # Apply trend regime multipliers
    for trend in ['strong', 'weak', 'no_trend']:
        mask = df['trend_regime'] == trend
        if mask.any():
            multiplier = trend_multipliers[trend]
            df.loc[mask, 'trend_component'] *= multiplier
            df.loc[mask, 'volume_component'] *= multiplier
    
    # Composite Alpha Construction
    # Combine signals with regime-aware weighting
    df['composite_signal'] = (
        df['volatility_component'] * 0.4 +
        df['trend_component'] * 0.4 +
        df['volume_component'] * 0.2
    )
    
    # Regime Transition Enhancement
    df['regime_change'] = (df['volatility_regime'] != df['volatility_regime'].shift(1)) | \
                         (df['trend_regime'] != df['trend_regime'].shift(1))
    
    # Amplify signals during regime transitions
    transition_multiplier = np.where(df['regime_change'], 1.5, 1.0)
    df['enhanced_signal'] = df['composite_signal'] * transition_multiplier
    
    # Smooth within stable regimes
    stable_mask = ~df['regime_change']
    df.loc[stable_mask, 'enhanced_signal'] = df.loc[stable_mask, 'enhanced_signal'].rolling(window=3).mean()
    
    # Final Alpha Output
    alpha = df['enhanced_signal'].fillna(0)
    
    return alpha
