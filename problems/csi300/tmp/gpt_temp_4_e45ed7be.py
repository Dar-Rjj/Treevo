import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Price-Volume Asymmetry with Regime-Dependent Momentum factor
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate returns and price direction
    df['returns'] = df['close'].pct_change()
    df['price_direction'] = np.where(df['returns'] > 0, 1, np.where(df['returns'] < 0, -1, 0))
    
    # Directional Volume Imbalance
    for i in range(len(df)):
        if i < 4:  # Need at least 4 days of history
            continue
            
        current_data = df.iloc[i]
        window_data = df.iloc[i-4:i+1]  # t-4 to t
        
        # Up-day and down-day filtering
        up_days = window_data[window_data['price_direction'] == 1]
        down_days = window_data[window_data['price_direction'] == -1]
        
        # Up-Day Volume Concentration
        if len(up_days) > 0:
            avg_up_volume = up_days['volume'].mean()
            up_volume_concentration = current_data['volume'] / avg_up_volume if avg_up_volume > 0 else 1
        else:
            up_volume_concentration = 1
            
        # Down-Day Volume Intensity
        if len(down_days) > 0:
            avg_down_volume = down_days['volume'].mean()
            down_volume_intensity = current_data['volume'] / avg_down_volume if avg_down_volume > 0 else 1
        else:
            down_volume_intensity = 1
            
        # Volume Asymmetry Index
        if len(up_days) > 0 and len(down_days) > 0:
            volume_asymmetry = (up_days['volume'].sum() / len(up_days)) / (down_days['volume'].sum() / len(down_days))
        else:
            volume_asymmetry = 1
            
        # Multi-Timeframe Divergence Detection
        short_term_divergence = []
        medium_term_divergence = []
        long_term_divergence = []
        
        for j in range(max(0, i-9), i+1):  # Look back up to 10 days
            if j < 1:
                continue
                
            current_window = df.iloc[max(0, j-1):j+1]
            
            # Price direction vs volume trend
            price_dir = current_window['price_direction'].iloc[-1]
            volume_trend = current_window['volume'].pct_change().iloc[-1]
            
            divergence_score = 0
            if price_dir > 0 and volume_trend < -0.05:  # Price up but volume declining
                divergence_score = -1
            elif price_dir < 0 and volume_trend > 0.05:  # Price down but volume increasing
                divergence_score = 1
            elif price_dir > 0 and volume_trend > 0.05:  # Confirmation
                divergence_score = 0.5
            elif price_dir < 0 and volume_trend < -0.05:  # Confirmation
                divergence_score = -0.5
                
            # Classify by timeframe
            if j >= i-2:  # 1-3 days
                short_term_divergence.append(divergence_score)
            elif j >= i-4:  # 3-5 days
                medium_term_divergence.append(divergence_score)
            elif j >= i-9:  # 5-10 days
                long_term_divergence.append(divergence_score)
        
        # Calculate divergence scores
        short_div_score = np.mean(short_term_divergence) if short_term_divergence else 0
        medium_div_score = np.mean(medium_term_divergence) if medium_term_divergence else 0
        long_div_score = np.mean(long_term_divergence) if long_term_divergence else 0
        
        # Transaction-Size Asymmetry (using amount as proxy for trade size)
        if i >= 4:
            recent_data = df.iloc[i-4:i+1]
            up_days_recent = recent_data[recent_data['price_direction'] == 1]
            down_days_recent = recent_data[recent_data['price_direction'] == -1]
            
            # Large Trade Direction Bias
            if len(up_days_recent) > 0 and len(down_days_recent) > 0:
                avg_amount_up = up_days_recent['amount'].mean()
                avg_amount_down = down_days_recent['amount'].mean()
                large_trade_bias = avg_amount_up / avg_amount_down if avg_amount_down > 0 else 1
            else:
                large_trade_bias = 1
        else:
            large_trade_bias = 1
            
        # Asymmetry-Momentum Integration
        if i >= 9:
            # Asymmetry Trend Strength
            recent_divergence = [short_div_score, medium_div_score, long_div_score]
            asymmetry_trend = np.polyfit(range(len(recent_divergence)), recent_divergence, 1)[0]
            
            # Extreme Asymmetry Points
            extreme_asymmetry = max(abs(short_div_score), abs(medium_div_score), abs(long_div_score))
            
            # Multi-Timeframe Asymmetry Convergence
            asymmetry_convergence = np.std([short_div_score, medium_div_score, long_div_score])
        else:
            asymmetry_trend = 0
            extreme_asymmetry = 0
            asymmetry_convergence = 1
            
        # Regime-Dependent Momentum
        if i >= 19:  # Need 20 days for volatility calculation
            # Volatility-Based Regimes
            volatility_20d = df['returns'].iloc[i-19:i+1].std()
            high_vol_regime = 1 if volatility_20d > df['returns'].iloc[:i+1].std() else 0
            
            # Trend-Based Regimes
            price_trend_20d = np.polyfit(range(20), df['close'].iloc[i-19:i+1].values, 1)[0]
            trend_strength = abs(price_trend_20d) / df['close'].iloc[i-19:i+1].std()
            trending_regime = 1 if trend_strength > 0.1 else 0
            
            # Dynamic Alpha Synthesis with regime weights
            if high_vol_regime and trending_regime:
                regime_weight = 1.2  # High momentum in trending high-vol regime
            elif high_vol_regime:
                regime_weight = 0.8  # Cautious in high-vol range-bound
            elif trending_regime:
                regime_weight = 1.1  # Confident in trending low-vol
            else:
                regime_weight = 0.9  # Neutral in range-bound low-vol
        else:
            regime_weight = 1.0
            
        # Final factor calculation
        volume_imbalance_component = (up_volume_concentration - down_volume_intensity) * volume_asymmetry
        divergence_component = (short_div_score * 0.4 + medium_div_score * 0.35 + long_div_score * 0.25)
        transaction_component = np.log(large_trade_bias) if large_trade_bias > 0 else 0
        asymmetry_component = asymmetry_trend * (1 - asymmetry_convergence) * extreme_asymmetry
        
        final_factor = regime_weight * (
            volume_imbalance_component * 0.3 +
            divergence_component * 0.3 +
            transaction_component * 0.2 +
            asymmetry_component * 0.2
        )
        
        result.iloc[i] = final_factor
    
    # Fill early NaN values with 0
    result = result.fillna(0)
    
    # Clean up temporary columns
    if 'returns' in df.columns:
        df.drop('returns', axis=1, inplace=True)
    if 'price_direction' in df.columns:
        df.drop('price_direction', axis=1, inplace=True)
    
    return result
