import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum-Volume Convergence Factor
    Combines price momentum across multiple timeframes with volume divergence analysis
    """
    data = df.copy()
    
    # Price Momentum Component
    data['momentum_3d'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_8d'] = data['close'] / data['close'].shift(8) - 1
    data['momentum_21d'] = data['close'] / data['close'].shift(21) - 1
    
    # Volume Divergence Component
    data['volume_ma_5d'] = data['volume'].rolling(window=5).mean()
    data['volume_ma_10d'] = data['volume'].rolling(window=10).mean()
    data['volume_divergence'] = data['volume_ma_5d'] / data['volume_ma_10d'] - 1
    
    # Volume Acceleration Adjustment
    data['volume_momentum_3d'] = data['volume'] / data['volume'].shift(3) - 1
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(3)) / (data['volume'].shift(3) / data['volume'].shift(6)) - 1
    
    # Market Condition Adaptation
    data['volatility_5d'] = data['close'].pct_change().rolling(window=5).std()
    data['volatility_15d_median'] = data['close'].pct_change().rolling(window=15).std().rolling(window=15).median()
    data['volume_ma_20d'] = data['volume'].rolling(window=20).mean()
    
    # Risk Management & Scaling
    data['cumulative_return_5d'] = (data['close'] / data['close'].shift(5) - 1)
    data['high_10d'] = data['high'].rolling(window=10).max()
    
    # Calculate True Range for volatility scaling
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    data['atr_5d'] = data['true_range'].rolling(window=5).mean()
    
    # Initialize factor
    factor_values = []
    
    for i in range(len(data)):
        if i < 21:  # Need enough data for all calculations
            factor_values.append(0.0)
            continue
            
        current = data.iloc[i]
        
        # Progressive momentum weighting (base weights)
        short_weight = 0.4
        medium_weight = 0.35
        long_weight = 0.25
        
        # Volatility-based timeframe emphasis
        if current['volatility_5d'] > current['volatility_15d_median']:
            # High volatility - emphasize short-term
            short_weight = 0.5
            medium_weight = 0.3
            long_weight = 0.2
        elif current['volatility_5d'] < current['volatility_15d_median'] * 0.7:
            # Low volatility - emphasize long-term
            short_weight = 0.3
            medium_weight = 0.3
            long_weight = 0.4
        
        # Weighted momentum combination
        weighted_momentum = (
            current['momentum_3d'] * short_weight +
            current['momentum_8d'] * medium_weight +
            current['momentum_21d'] * long_weight
        )
        
        # Volume divergence multiplier
        volume_multiplier = 1.0 + current['volume_divergence']
        
        # Volume acceleration adjustment
        if current['volume_acceleration'] > 0.1:
            volume_multiplier *= 1.2
        elif current['volume_acceleration'] < -0.1:
            volume_multiplier *= 0.8
        
        # Volume reliability assessment
        volume_influence = 1.0
        if current['volume'] > current['volume_ma_20d'] * 1.4:
            volume_influence = 1.3
        elif current['volume'] < current['volume_ma_20d'] * 0.8:
            volume_influence = 0.7
        
        # Convergence Detection Logic
        convergence_multiplier = 1.0
        
        # Momentum direction analysis
        momentum_directions = [
            np.sign(current['momentum_3d']),
            np.sign(current['momentum_8d']),
            np.sign(current['momentum_21d'])
        ]
        
        positive_count = sum(1 for d in momentum_directions if d > 0)
        negative_count = sum(1 for d in momentum_directions if d < 0)
        
        # Strong convergence
        if (positive_count == 3 or negative_count == 3) and current['volume_divergence'] > 0:
            convergence_multiplier = 1.5
        # Weak convergence
        elif (positive_count >= 2 or negative_count >= 2) and (
            (current['volume_divergence'] > 0 and positive_count >= 2) or
            (current['volume_divergence'] < 0 and negative_count >= 2)
        ):
            convergence_multiplier = 1.0
        # Divergence
        elif ((positive_count >= 2 and current['volume_divergence'] < 0) or
              (negative_count >= 2 and current['volume_divergence'] > 0)):
            convergence_multiplier = 0.5
        
        # Momentum Persistence Weighting
        if positive_count == 3 or negative_count == 3:
            convergence_multiplier *= 1.3
        elif positive_count == 2 or negative_count == 2:
            convergence_multiplier *= 1.1
        
        # Volume consistency
        if current['volume_divergence'] > 0.1 and positive_count >= 2:
            convergence_multiplier *= 1.2
        elif current['volume_divergence'] < -0.1 and negative_count >= 2:
            convergence_multiplier *= 1.2
        elif (positive_count > 0 and negative_count > 0):
            convergence_multiplier *= 0.8
        
        # Risk Management & Scaling
        risk_multiplier = 1.0
        if current['cumulative_return_5d'] < -0.08:
            risk_multiplier *= 0.6
        
        # Check for 3-day consecutive declines
        if i >= 3:
            recent_returns = [data.iloc[i-j]['close'] / data.iloc[i-j-1]['close'] - 1 for j in range(1, 4)]
            if all(r < 0 for r in recent_returns):
                risk_multiplier *= 0.8
        
        # Recent breakout
        if current['close'] > current['high_10d']:
            risk_multiplier *= 1.2
        
        # Combine all components
        base_factor = weighted_momentum * volume_multiplier * volume_influence
        adjusted_factor = base_factor * convergence_multiplier * risk_multiplier
        
        # Volatility scaling
        if current['atr_5d'] > 0:
            volatility_scaled_factor = adjusted_factor / current['atr_5d']
        else:
            volatility_scaled_factor = adjusted_factor
        
        factor_values.append(volatility_scaled_factor)
    
    factor_series = pd.Series(factor_values, index=data.index, name='momentum_volume_convergence')
    return factor_series
