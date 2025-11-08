import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate True Range
    data['prev_close'] = data['close'].shift(1)
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['prev_close'])
    data['tr3'] = abs(data['low'] - data['prev_close'])
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Volatility Environment Assessment
    data['atr_5'] = data['true_range'].rolling(window=5, min_periods=5).mean()
    data['atr_20'] = data['true_range'].rolling(window=20, min_periods=20).mean()
    data['vol_regime'] = data['atr_5'] / data['atr_20']
    
    # Momentum-Persistence Component
    # Price momentum strength
    def calc_slope(series):
        if len(series) < 5:
            return np.nan
        x = np.arange(5)
        y = series.values
        return np.polyfit(x, y, 1)[0]
    
    data['price_slope'] = data['close'].rolling(window=5, min_periods=5).apply(calc_slope, raw=False)
    
    # Momentum consistency (consecutive same-direction moves)
    def count_consecutive_direction(series):
        if len(series) < 5:
            return np.nan
        changes = np.diff(series)
        signs = np.sign(changes)
        if len(signs) == 0:
            return 0
        current_sign = signs[-1]
        count = 1
        for i in range(len(signs)-2, -1, -1):
            if signs[i] == current_sign:
                count += 1
            else:
                break
        return count
    
    data['direction_count'] = data['close'].rolling(window=5, min_periods=5).apply(count_consecutive_direction, raw=False)
    data['price_persistence'] = data['price_slope'] * data['direction_count']
    
    # Volume momentum confirmation
    data['volume_slope'] = data['volume'].rolling(window=5, min_periods=5).apply(calc_slope, raw=False)
    
    # Volume-price alignment
    def volume_alignment(row):
        if pd.isna(row['price_slope']) or pd.isna(row['volume_slope']):
            return np.nan
        if np.sign(row['price_slope']) == np.sign(row['volume_slope']):
            return abs(row['volume_slope'])
        else:
            return -abs(row['volume_slope'])
    
    data['volume_alignment'] = data.apply(volume_alignment, axis=1)
    data['momentum_component'] = data['price_persistence'] * data['volume_alignment']
    
    # Microstructure Efficiency
    # Intraday rejection signals
    data['upper_shadow'] = (data['high'] - data[['open', 'close']].max(axis=1)) / (data['high'] - data['low']).replace(0, np.nan)
    data['lower_shadow'] = (data[['open', 'close']].min(axis=1) - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Net efficiency signal
    data['efficiency_signal'] = data['lower_shadow'] - data['upper_shadow']
    data['efficiency_change'] = data['efficiency_signal'] - data['efficiency_signal'].shift(1)
    
    # Generate Composite Alpha Signal
    data['composite_signal'] = data['momentum_component'] * data['efficiency_change'] * data['volume']
    
    # Apply Volatility-Regime Adjustment
    data['final_alpha'] = data['composite_signal'] / data['vol_regime'].replace(0, np.nan)
    
    # Return the final alpha factor series
    return data['final_alpha']
