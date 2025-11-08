import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel alpha factor combining price-volume momentum, volatility normalization,
    and regime-aware scaling for predicting future stock returns.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Parameters
    short_term = 5
    medium_term = 20
    long_term = 60
    vol_lookback = 20
    regime_lookback = 60
    
    # Initialize result series
    factor_values = pd.Series(index=df.index, dtype=float)
    
    for i in range(max(long_term, regime_lookback), len(df)):
        current_data = df.iloc[:i+1]
        
        # 1. Volume-Weighted Price Momentum
        # Multi-day momentum with volume weighting
        close_prices = current_data['close'].values
        volumes = current_data['volume'].values
        
        # Calculate momentum over different periods
        mom_short = (close_prices[-1] / close_prices[-short_term] - 1) if i >= short_term else 0
        mom_medium = (close_prices[-1] / close_prices[-medium_term] - 1) if i >= medium_term else 0
        mom_long = (close_prices[-1] / close_prices[-long_term] - 1) if i >= long_term else 0
        
        # Volume weighting using average volume over momentum periods
        vol_weight_short = np.mean(volumes[-short_term:]) if i >= short_term else 1
        vol_weight_medium = np.mean(volumes[-medium_term:]) if i >= medium_term else 1
        vol_weight_long = np.mean(volumes[-long_term:]) if i >= long_term else 1
        
        volume_weighted_momentum = (
            mom_short * vol_weight_short * 0.5 +
            mom_medium * vol_weight_medium * 0.3 +
            mom_long * vol_weight_long * 0.2
        )
        
        # 2. Volume-Adjusted Range Efficiency
        # Calculate true range and volume-scaled efficiency
        high_prices = current_data['high'].values
        low_prices = current_data['low'].values
        
        if i >= 1:
            true_ranges = []
            volume_scaled_ranges = []
            
            for j in range(max(1, i-20), i+1):
                prev_close = close_prices[j-1] if j > 0 else close_prices[j]
                true_range = max(
                    high_prices[j] - low_prices[j],
                    abs(high_prices[j] - prev_close),
                    abs(low_prices[j] - prev_close)
                )
                true_ranges.append(true_range)
                
                # Volume-scaled range
                if volumes[j] > 0:
                    volume_scaled_range = true_range / volumes[j]
                    volume_scaled_ranges.append(volume_scaled_range)
            
            if len(true_ranges) > 0 and len(volume_scaled_ranges) > 0:
                avg_true_range = np.mean(true_ranges)
                avg_volume_scaled_range = np.mean(volume_scaled_ranges)
                
                # Efficiency ratio: price movement vs range
                price_movement = abs(close_prices[-1] - close_prices[-len(true_ranges)])
                total_range = sum(true_ranges)
                
                range_efficiency = price_movement / total_range if total_range > 0 else 0
                volume_adjusted_efficiency = range_efficiency * (1 / (avg_volume_scaled_range + 1e-8))
            else:
                volume_adjusted_efficiency = 0
        else:
            volume_adjusted_efficiency = 0
        
        # 3. Decay-Adjusted Multi-Horizon Signals
        # Exponential decay weighting for different horizons
        decay_weights = {
            'short': 0.6,
            'medium': 0.3,
            'long': 0.1
        }
        
        decay_adjusted_signal = (
            mom_short * decay_weights['short'] +
            mom_medium * decay_weights['medium'] +
            mom_long * decay_weights['long']
        )
        
        # 4. Volatility-Normalized Factors
        # Calculate daily volatility using high-low range
        if i >= vol_lookback:
            recent_highs = high_prices[-vol_lookback:]
            recent_lows = low_prices[-vol_lookback:]
            daily_ranges = recent_highs - recent_lows
            avg_daily_range = np.mean(daily_ranges)
            range_volatility = np.std(daily_ranges) / (avg_daily_range + 1e-8) if avg_daily_range > 0 else 0
            
            # Volatility-adjusted momentum
            volatility_adjusted_momentum = volume_weighted_momentum / (range_volatility + 1e-8)
        else:
            volatility_adjusted_momentum = volume_weighted_momentum
        
        # 5. Volume-Volatility Relationship
        if i >= vol_lookback:
            recent_volumes = volumes[-vol_lookback:]
            volume_volatility = np.std(recent_volumes) / (np.mean(recent_volumes) + 1e-8)
            
            # Price volatility (using close returns)
            recent_returns = np.diff(close_prices[-vol_lookback:]) / close_prices[-vol_lookback:-1]
            price_volatility = np.std(recent_returns) if len(recent_returns) > 0 else 0
            
            volume_volatility_ratio = volume_volatility / (price_volatility + 1e-8)
        else:
            volume_volatility_ratio = 1
        
        # 6. Regime-Aware Scaling
        # Detect volatility regimes
        if i >= regime_lookback:
            regime_returns = np.diff(close_prices[-regime_lookback:]) / close_prices[-regime_lookback:-1]
            regime_volatility = np.std(regime_returns) if len(regime_returns) > 0 else 0
            
            # Historical volatility for comparison
            if i >= regime_lookback * 2:
                historical_returns = np.diff(close_prices[-regime_lookback*2:-regime_lookback]) / close_prices[-regime_lookback*2:-regime_lookback-1]
                historical_volatility = np.std(historical_returns) if len(historical_returns) > 0 else regime_volatility
            else:
                historical_volatility = regime_volatility
            
            # Volatility regime detection
            if regime_volatility > historical_volatility * 1.5:
                regime_multiplier = 0.7  # Reduce sensitivity in high volatility
            elif regime_volatility < historical_volatility * 0.7:
                regime_multiplier = 1.3  # Increase sensitivity in low volatility
            else:
                regime_multiplier = 1.0  # Normal regime
        else:
            regime_multiplier = 1.0
        
        # 7. Combine all components with economic interpretation
        # Trend identification component
        trend_component = volume_weighted_momentum * volume_adjusted_efficiency
        
        # Mean reversion component (inverse relationship for overbought/oversold)
        mean_reversion_component = -decay_adjusted_signal * volume_volatility_ratio
        
        # Liquidity proxy component
        liquidity_component = volatility_adjusted_momentum
        
        # Final factor combination with regime-aware scaling
        final_factor = (
            trend_component * 0.4 +
            mean_reversion_component * 0.35 +
            liquidity_component * 0.25
        ) * regime_multiplier
        
        factor_values.iloc[i] = final_factor
    
    # Forward fill any NaN values
    factor_values = factor_values.ffill()
    
    return factor_values
