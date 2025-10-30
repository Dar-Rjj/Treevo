import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize factor series
    factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i < 21:  # Need enough data for calculations
            factor.iloc[i] = 0
            continue
            
        current_data = data.iloc[:i+1]
        
        # Volatility-Adjusted Gap Momentum
        gap_momentum = (current_data['open'].iloc[i] / current_data['close'].iloc[i-1] - 1) * \
                      (current_data['close'].iloc[i] / current_data['open'].iloc[i] - 1)
        
        vol_window = current_data.iloc[i-4:i+1]
        volatility_adj = (current_data['high'].iloc[i] - current_data['low'].iloc[i]) / \
                        (vol_window['high'] - vol_window['low']).mean()
        
        vol_adj_gap_momentum = gap_momentum / volatility_adj if volatility_adj != 0 else 0
        
        # Volume-Price Efficiency Ratio
        price_efficiency = abs(current_data['close'].iloc[i] - current_data['open'].iloc[i]) / \
                          (current_data['high'].iloc[i] - current_data['low'].iloc[i]) if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) != 0 else 0
        
        volume_window = current_data.iloc[i-4:i+1]
        volume_efficiency = current_data['volume'].iloc[i] / volume_window['volume'].mean()
        
        price_change = current_data['close'].iloc[i] / current_data['close'].iloc[i-1] - 1
        
        volume_price_efficiency = price_efficiency * volume_efficiency * price_change
        
        # Multi-Timeframe Pressure Index
        short_term_pressure = (current_data['high'].iloc[i] - max(current_data['open'].iloc[i], current_data['close'].iloc[i])) / \
                             (current_data['high'].iloc[i] - current_data['low'].iloc[i]) if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) != 0 else 0
        
        medium_window = current_data.iloc[i-4:i+1]
        medium_high = medium_window['high'].max()
        medium_open_close_max = max(medium_window['open'].max(), medium_window['close'].max())
        medium_low = medium_window['low'].min()
        
        medium_term_pressure = (medium_high - medium_open_close_max) / (medium_high - medium_low) if (medium_high - medium_low) != 0 else 0
        
        pressure_index = short_term_pressure * medium_term_pressure * current_data['volume'].iloc[i]
        
        # Amount-Volume Divergence Factor
        price_level = (current_data['high'].iloc[i] + current_data['low'].iloc[i] + current_data['close'].iloc[i]) / 3
        volume_weighted_price = current_data['amount'].iloc[i] / current_data['volume'].iloc[i] if current_data['volume'].iloc[i] != 0 else price_level
        
        divergence = abs(volume_weighted_price - price_level) / (current_data['high'].iloc[i] - current_data['low'].iloc[i]) if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) != 0 else 0
        
        volume_change = current_data['volume'].iloc[i] / current_data['volume'].iloc[i-1] - 1 if current_data['volume'].iloc[i-1] != 0 else 0
        
        amount_volume_divergence = volume_change / divergence if divergence != 0 else 0
        
        # Regime-Enhanced Reversal
        trend_consistency = 0
        for lookback in [3, 8, 21]:
            if i >= lookback:
                current_sign = np.sign(current_data['close'].iloc[i] / current_data['close'].iloc[i-1] - 1)
                lookback_sign = np.sign(current_data['close'].iloc[i] / current_data['close'].iloc[i-lookback] - 1)
                if current_sign == lookback_sign:
                    trend_consistency += 1
        
        intraday_reversal = ((current_data['close'].iloc[i] - current_data['open'].iloc[i]) / 
                            (current_data['high'].iloc[i] - current_data['low'].iloc[i]) if (current_data['high'].iloc[i] - current_data['low'].iloc[i]) != 0 else 0) * \
                           (current_data['open'].iloc[i] / current_data['close'].iloc[i-1] - 1)
        
        regime_reversal = intraday_reversal * (1 + trend_consistency / 3)
        
        # Combine all factors (equal weighting for simplicity)
        combined_factor = (vol_adj_gap_momentum + volume_price_efficiency + pressure_index + 
                          amount_volume_divergence + regime_reversal) / 5
        
        factor.iloc[i] = combined_factor
    
    return factor
