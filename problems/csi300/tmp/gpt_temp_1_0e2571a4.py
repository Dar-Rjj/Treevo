import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Volume Divergence Strength alpha factor
    Detects and quantifies divergence between price momentum and volume momentum across multiple timeframes
    """
    # Extract price and volume data
    close = df['close']
    volume = df['volume']
    
    # Multi-period price momentum calculations
    mom_5d = close.pct_change(5)  # 5-day price momentum
    mom_10d = close.pct_change(10)  # 10-day price momentum
    mom_20d = close.pct_change(20)  # 20-day price momentum
    
    # Multi-period volume momentum calculations
    vol_mom_5d = volume.pct_change(5)  # 5-day volume momentum
    vol_mom_10d = volume.pct_change(10)  # 10-day volume momentum
    vol_mom_20d = volume.pct_change(20)  # 20-day volume momentum
    
    # Momentum-Volume product calculations
    mom_vol_product_5d = mom_5d * vol_mom_5d
    mom_vol_product_10d = mom_10d * vol_mom_10d
    mom_vol_product_20d = mom_20d * vol_mom_20d
    
    # Divergence strength quantification
    def calculate_divergence_strength(price_mom, vol_mom, product):
        # Initialize result array
        result = np.zeros(len(price_mom))
        
        for i in range(len(price_mom)):
            if pd.isna(price_mom.iloc[i]) or pd.isna(vol_mom.iloc[i]):
                result[i] = np.nan
                continue
                
            # Sign agreement analysis
            price_sign = np.sign(price_mom.iloc[i])
            vol_sign = np.sign(vol_mom.iloc[i])
            
            if price_sign == vol_sign:  # Confirming momentum
                # Positive weight for confirming momentum
                strength = np.sqrt(abs(product.iloc[i])) * price_sign
            elif price_sign != vol_sign and price_sign != 0 and vol_sign != 0:  # Divergence detected
                # Negative weight for opposing divergence
                strength = -np.sqrt(abs(product.iloc[i])) * price_sign
            else:  # Neutral cases
                strength = 0
                
            result[i] = strength
            
        return pd.Series(result, index=price_mom.index)
    
    # Calculate divergence strength for each period
    div_strength_5d = calculate_divergence_strength(mom_5d, vol_mom_5d, mom_vol_product_5d)
    div_strength_10d = calculate_divergence_strength(mom_10d, vol_mom_10d, mom_vol_product_10d)
    div_strength_20d = calculate_divergence_strength(mom_20d, vol_mom_20d, mom_vol_product_20d)
    
    # Multi-timeframe divergence analysis
    def get_divergence_pattern(div_5d, div_10d, div_20d):
        pattern_scores = np.zeros(len(div_5d))
        
        for i in range(len(div_5d)):
            if any(pd.isna([div_5d.iloc[i], div_10d.iloc[i], div_20d.iloc[i]])):
                pattern_scores[i] = np.nan
                continue
                
            # Count divergence signals (negative values indicate divergence)
            div_count = sum([1 for val in [div_5d.iloc[i], div_10d.iloc[i], div_20d.iloc[i]] if val < 0])
            conf_count = sum([1 for val in [div_5d.iloc[i], div_10d.iloc[i], div_20d.iloc[i]] if val > 0])
            
            if div_count == 3:  # Consistent divergence across all periods
                weight = 1.5
            elif div_count >= 2:  # Strong divergence
                weight = 1.2
            elif div_count == 1:  # Moderate divergence
                weight = 1.0
            elif conf_count >= 2:  # Confirming momentum
                weight = 0.8
            else:  # Neutral
                weight = 0.5
                
            # Weighted average of period-specific signals
            valid_vals = [val for val in [div_5d.iloc[i], div_10d.iloc[i], div_20d.iloc[i]] if not pd.isna(val)]
            if valid_vals:
                pattern_scores[i] = np.mean(valid_vals) * weight
            else:
                pattern_scores[i] = np.nan
                
        return pd.Series(pattern_scores, index=div_5d.index)
    
    # Generate final alpha factor
    alpha_factor = get_divergence_pattern(div_strength_5d, div_strength_10d, div_strength_20d)
    
    return alpha_factor
