import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Fractal Momentum Divergence factor combining:
    - Fractal dimension analysis using Hurst exponent
    - Multi-timeframe momentum divergence
    - Volume-confirmed regime transitions
    """
    
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    factor = pd.Series(index=df.index, dtype=float)
    
    # Minimum required data points for calculations
    min_periods = 20
    
    for i in range(min_periods, len(df)):
        current_data = df.iloc[:i+1]  # Only use current and past data
        
        # 1. Fractal Dimension Calculation using Hurst exponent
        if len(current_data) >= min_periods:
            # Calculate price changes
            price_series = current_data['close'].values
            log_returns = np.log(price_series[1:] / price_series[:-1])
            
            # Calculate Hurst exponent using R/S analysis
            lags = range(2, min(10, len(log_returns)))
            rs_values = []
            
            for lag in lags:
                if len(log_returns) >= lag:
                    # Split data into chunks of size lag
                    chunks = [log_returns[j:j+lag] for j in range(0, len(log_returns), lag)]
                    chunk_rs = []
                    
                    for chunk in chunks:
                        if len(chunk) >= 2:
                            # Calculate mean-adjusted series
                            mean_chunk = np.mean(chunk)
                            adjusted = chunk - mean_chunk
                            # Calculate cumulative deviations
                            cumulative = np.cumsum(adjusted)
                            # Calculate range
                            R = np.max(cumulative) - np.min(cumulative)
                            # Calculate standard deviation
                            S = np.std(chunk)
                            if S > 0:
                                chunk_rs.append(R / S)
                    
                    if chunk_rs:
                        rs_values.append(np.mean(chunk_rs))
            
            if len(rs_values) >= 2:
                # Fit log-log relationship to estimate Hurst exponent
                x = np.log(lags[:len(rs_values)])
                y = np.log(rs_values)
                hurst_exponent = np.polyfit(x, y, 1)[0]
            else:
                hurst_exponent = 0.5
        else:
            hurst_exponent = 0.5
        
        # 2. Multi-timeframe Momentum Calculations
        current_idx = current_data.index[-1]
        
        # Short-term momentum (2-day)
        if len(current_data) >= 3:
            st_momentum = (current_data['close'].iloc[-1] / current_data['close'].iloc[-3] - 1)
        else:
            st_momentum = 0
        
        # Medium-term momentum (8-day)
        if len(current_data) >= 9:
            mt_momentum = (current_data['close'].iloc[-1] / current_data['close'].iloc[-9] - 1)
        else:
            mt_momentum = 0
        
        # 3. Volume Cluster Analysis
        if len(current_data) >= 10:
            # Calculate rolling volume statistics
            recent_volume = current_data['volume'].iloc[-10:]
            volume_mean = recent_volume.mean()
            volume_std = recent_volume.std()
            
            if volume_std > 0:
                current_volume_z = (current_data['volume'].iloc[-1] - volume_mean) / volume_std
            else:
                current_volume_z = 0
            
            # Volume regime classification
            volume_regime = 1 if current_volume_z > 1 else (-1 if current_volume_z < -1 else 0)
        else:
            volume_regime = 0
        
        # 4. Range Efficiency Calculation
        if len(current_data) >= 5:
            recent_data = current_data.iloc[-5:]
            daily_ranges = (recent_data['high'] - recent_data['low']) / recent_data['close'].shift(1)
            range_efficiency = np.mean(daily_ranges.dropna())
            
            # Asymmetric range calculation
            up_moves = (recent_data['high'] - recent_data['open']) / recent_data['open']
            down_moves = (recent_data['open'] - recent_data['low']) / recent_data['open']
            range_asymmetry = np.mean(up_moves.dropna()) - np.mean(down_moves.dropna())
        else:
            range_efficiency = 0
            range_asymmetry = 0
        
        # 5. Momentum-Fractal Divergence Signal
        # Fractal persistence indicates trending (H > 0.5) or mean-reverting (H < 0.5) behavior
        fractal_trend_strength = hurst_exponent - 0.5
        
        # Momentum divergence: when momentum direction contradicts fractal structure
        momentum_divergence = 0
        if fractal_trend_strength > 0.1:  # Trending market
            if st_momentum < 0 and mt_momentum < 0:  # Contradictory momentum
                momentum_divergence = -1
        elif fractal_trend_strength < -0.1:  # Mean-reverting market
            if st_momentum > 0 and mt_momentum > 0:  # Contradictory momentum
                momentum_divergence = 1
        
        # 6. Volume-Confirmed Regime Signal
        volume_signal = volume_regime * (st_momentum + mt_momentum)
        
        # 7. Range Efficiency Signal
        range_signal = range_asymmetry * range_efficiency
        
        # 8. Combined Factor Calculation
        # Weight the components based on their predictive strength
        factor_value = (
            0.4 * momentum_divergence +
            0.3 * volume_signal +
            0.3 * range_signal
        )
        
        factor.loc[current_idx] = factor_value
    
    # Fill initial NaN values with 0
    factor = factor.fillna(0)
    
    return factor
