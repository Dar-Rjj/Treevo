import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Gaps
    data['ultra_short_gap'] = data['open'] / data['close'].shift(2) - 1
    data['short_gap'] = data['open'] / data['close'].shift(5) - 1
    data['medium_gap'] = data['open'] / data['close'].shift(15) - 1
    
    # Gap Persistence
    data['gap_holding_ratio'] = (data['close'] - data['open']) / (data['open'] - data['close'].shift(1) + 1e-8)
    data['gap_momentum_signal'] = data['short_gap'] * data['gap_holding_ratio']
    
    # Volume Patterns
    # Volume Concentration (Sum of top 3 volumes in last 10 days / Total 10-day volume)
    def volume_concentration(series):
        if len(series) < 10:
            return np.nan
        top_3_sum = series.nlargest(3).sum()
        total_sum = series.sum()
        return top_3_sum / total_sum if total_sum > 0 else 0
    
    data['volume_concentration'] = data['volume'].rolling(window=10, min_periods=10).apply(volume_concentration, raw=False)
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    
    # Price Efficiency
    data['price_efficiency'] = np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Net Pressure Volume (simplified as volume * sign of price change)
    data['net_pressure_volume'] = data['volume'] * np.sign(data['close'] - data['open'])
    data['volume_price_alignment'] = (data['net_pressure_volume'] / (data['volume'] + 1e-8)) * data['price_efficiency']
    
    # Amount Confirmation
    data['short_amount_momentum'] = data['amount'] / data['amount'].shift(5) - 1
    data['amount_gap_alignment'] = data['short_amount_momentum'] * data['short_gap']
    
    # Amount-Volume Consistency
    data['amount_pressure'] = data['amount'] * data['price_efficiency']
    
    # Range Adjustment
    # True Range Components
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    data['avg_true_range_5d'] = data['true_range'].rolling(window=5, min_periods=5).mean()
    
    # Range-Adjusted Signals
    data['gap_range_ratio'] = data['gap_momentum_signal'] / (data['avg_true_range_5d'] + 1e-8)
    data['volume_efficiency_range'] = data['volume_price_alignment'] / (data['avg_true_range_5d'] + 1e-8)
    
    # Composite Alpha
    # Core Components
    data['base_signal'] = data['gap_momentum_signal'] * data['gap_holding_ratio']
    data['amount_multiplier'] = 1 + np.abs(data['amount_gap_alignment'])
    data['volume_adjustment'] = data['volume_concentration'] * data['volume_price_alignment']
    
    # Final Integration
    data['weighted_factor'] = data['base_signal'] * data['amount_multiplier'] * data['volume_adjustment']
    data['final_alpha'] = data['weighted_factor'] / (data['avg_true_range_5d'] + 1e-8)
    
    # Return the final alpha factor series
    return data['final_alpha']
