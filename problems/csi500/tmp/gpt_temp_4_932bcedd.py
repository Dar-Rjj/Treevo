import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining geometric momentum acceleration, volatility-scaled geometric momentum,
    volume-weighted geometric efficiency, and acceleration-scaled volatility regime.
    """
    close = df['close']
    high = df['high']
    low = df['low']
    open_price = df['open']
    volume = df['volume']
    
    # Calculate returns for volatility calculations
    returns = close.pct_change()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(11, len(df)):
        current_idx = df.index[i]
        
        try:
            # 1. Geometric Momentum Acceleration
            # Multi-Timeframe Momentum
            mom_2d = close.iloc[i] / close.iloc[i-2] - 1
            mom_5d = close.iloc[i] / close.iloc[i-5] - 1
            mom_10d = close.iloc[i] / close.iloc[i-10] - 1
            
            # Momentum Acceleration
            mom_2d_prev = close.iloc[i-1] / close.iloc[i-3] - 1
            mom_5d_prev = close.iloc[i-1] / close.iloc[i-6] - 1
            mom_10d_prev = close.iloc[i-1] / close.iloc[i-11] - 1
            
            acc_2d = mom_2d - mom_2d_prev
            acc_5d = mom_5d - mom_5d_prev
            acc_10d = mom_10d - mom_10d_prev
            
            # Combined Factor
            geo_mom = (mom_2d * mom_5d * mom_10d) ** (1/3)
            geo_acc = (acc_2d * acc_5d * acc_10d) ** (1/3)
            factor1 = geo_mom * geo_acc
            
            # 2. Volatility-Scaled Geometric Momentum
            # Volatility Components
            vol_2d = returns.iloc[i-1:i+1].std() if i >= 1 else np.nan
            vol_5d = returns.iloc[i-4:i+1].std() if i >= 4 else np.nan
            vol_10d = returns.iloc[i-9:i+1].std() if i >= 9 else np.nan
            
            # Combined Factor
            geo_vol = (vol_2d * vol_5d * vol_10d) ** (1/3)
            factor2 = geo_mom / geo_vol if geo_vol != 0 else 0
            
            # 3. Volume-Weighted Geometric Efficiency
            # Efficiency Components
            opening_eff = (close.iloc[i] - open_price.iloc[i]) / (high.iloc[i] - low.iloc[i]) if (high.iloc[i] - low.iloc[i]) != 0 else 0
            high_low_eff = (close.iloc[i] - low.iloc[i]) / (high.iloc[i] - low.iloc[i]) if (high.iloc[i] - low.iloc[i]) != 0 else 0
            price_persistence = (close.iloc[i] - close.iloc[i-1]) / (high.iloc[i] - low.iloc[i]) if (high.iloc[i] - low.iloc[i]) != 0 else 0
            
            # Volume Components
            vol_ratio_2d = volume.iloc[i] / volume.iloc[i-2] if volume.iloc[i-2] != 0 else 1
            vol_ratio_5d = volume.iloc[i] / volume.iloc[i-5] if volume.iloc[i-5] != 0 else 1
            vol_ratio_10d = volume.iloc[i] / volume.iloc[i-10] if volume.iloc[i-10] != 0 else 1
            
            # Combined Factor
            geo_eff = (opening_eff * high_low_eff * price_persistence) ** (1/3)
            geo_vol_ratio = (vol_ratio_2d * vol_ratio_5d * vol_ratio_10d) ** (1/3)
            factor3 = geo_eff * geo_vol_ratio
            
            # 4. Acceleration-Scaled Volatility Regime
            # Acceleration Components
            acc_2d_price = mom_2d - (close.iloc[i-1] / close.iloc[i-3] - 1)
            acc_5d_price = mom_5d - (close.iloc[i-1] / close.iloc[i-6] - 1)
            acc_10d_price = mom_10d - (close.iloc[i-1] / close.iloc[i-11] - 1)
            
            # Combined Factor
            geo_acc_price = (acc_2d_price * acc_5d_price * acc_10d_price) ** (1/3)
            factor4 = geo_acc_price / geo_vol if geo_vol != 0 else 0
            
            # Combine all factors (equal weighting)
            combined_factor = (factor1 + factor2 + factor3 + factor4) / 4
            result.iloc[i] = combined_factor
            
        except (IndexError, KeyError, ZeroDivisionError):
            result.iloc[i] = np.nan
    
    return result
