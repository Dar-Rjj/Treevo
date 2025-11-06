import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Adaptive Momentum Acceleration factor
    Combines multi-timeframe momentum acceleration with volume confirmation
    and volatility regime adaptation
    """
    
    # Copy data to avoid modifying original
    data = df.copy()
    
    # Calculate True Range for volatility assessment
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility regime classification using rolling standard deviation of true range
    data['volatility_regime'] = data['true_range'].rolling(window=20, min_periods=10).std()
    
    # Define regime-dependent parameters
    high_vol_threshold = data['volatility_regime'].quantile(0.7)
    low_vol_threshold = data['volatility_regime'].quantile(0.3)
    
    # Multi-timeframe momentum calculations
    # Short-term momentum (1-3 days)
    data['momentum_1d'] = data['close'].pct_change(1)
    data['momentum_2d'] = data['close'].pct_change(2)
    data['momentum_3d'] = data['close'].pct_change(3)
    data['short_term_momentum'] = (data['momentum_1d'] + data['momentum_2d'] + data['momentum_3d']) / 3
    
    # Medium-term momentum (5-10 days) using linear regression slopes
    def rolling_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series)):
            if i < window - 1:
                slopes.append(np.nan)
            else:
                y = series.iloc[i-window+1:i+1].values
                if len(y) == window:
                    slope = np.polyfit(x, y, 1)[0] / y[0]  # Normalized slope
                    slopes.append(slope)
                else:
                    slopes.append(np.nan)
        return pd.Series(slopes, index=series.index)
    
    data['medium_term_momentum_5d'] = rolling_slope(data['close'], 5)
    data['medium_term_momentum_10d'] = rolling_slope(data['close'], 10)
    data['medium_term_momentum'] = (data['medium_term_momentum_5d'] + data['medium_term_momentum_10d']) / 2
    
    # Momentum acceleration signals
    data['momentum_acceleration'] = data['short_term_momentum'] - data['medium_term_momentum']
    
    # Volume trend analysis
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_ma_10'] = data['volume'].rolling(window=10, min_periods=5).mean()
    data['volume_trend'] = data['volume_ma_5'] / data['volume_ma_10'] - 1
    
    # Volume concentration during momentum shifts
    data['momentum_change'] = data['momentum_acceleration'].diff()
    data['volume_momentum_alignment'] = np.where(
        data['momentum_change'] * data['volume_trend'] > 0,
        abs(data['momentum_change']) * data['volume_trend'],
        0
    )
    
    # Adaptive signal generation based on volatility regime
    def adaptive_momentum_acceleration(row):
        if pd.isna(row['volatility_regime']):
            return np.nan
        
        if row['volatility_regime'] > high_vol_threshold:
            # High volatility: use shorter lookbacks and emphasize recent momentum
            weight_short = 0.7
            weight_medium = 0.3
            volume_weight = 0.4
        elif row['volatility_regime'] < low_vol_threshold:
            # Low volatility: use balanced lookbacks and emphasize medium-term trends
            weight_short = 0.4
            weight_medium = 0.6
            volume_weight = 0.6
        else:
            # Normal volatility: balanced approach
            weight_short = 0.5
            weight_medium = 0.5
            volume_weight = 0.5
        
        momentum_component = (weight_short * row['short_term_momentum'] + 
                            weight_medium * row['medium_term_momentum'])
        
        volume_component = volume_weight * row['volume_momentum_alignment']
        
        return momentum_component + volume_component
    
    # Calculate final factor
    data['factor'] = data.apply(adaptive_momentum_acceleration, axis=1)
    
    # Clean up intermediate columns
    result = data['factor'].copy()
    
    return result
