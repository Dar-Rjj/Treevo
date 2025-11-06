import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Pressure Asymmetry Analysis
    # Bull-Bear Pressure
    bull_bear_pressure = (data['close'] - data['low']) / (data['high'] - data['close'])
    bull_bear_pressure = bull_bear_pressure.replace([np.inf, -np.inf], np.nan)
    
    # Gap Momentum
    gap_momentum = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Intraday Pressure
    intraday_pressure = (data['close'] - data['open']) / (data['high'] - data['low'])
    intraday_pressure = intraday_pressure.replace([np.inf, -np.inf], np.nan)
    
    # Fractal Efficiency Assessment
    # Short-Term Efficiency (5-day)
    close_ratio_5d = data['close'] / data['close'].shift(5) - 1
    
    # Calculate rolling high and low for 5-day window (t-4 to t)
    rolling_high_5d = data['high'].rolling(window=5, min_periods=5).max()
    rolling_low_5d = data['low'].rolling(window=5, min_periods=5).min()
    price_range_5d = (rolling_high_5d - rolling_low_5d) / data['close']
    
    short_term_efficiency = close_ratio_5d / price_range_5d
    short_term_efficiency = short_term_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Medium-Term Efficiency (20-day)
    close_ratio_20d = data['close'] / data['close'].shift(20) - 1
    
    # Calculate sum of absolute price changes for 20-day window
    abs_price_changes = abs(data['close'] - data['close'].shift(1))
    sum_abs_changes_20d = abs_price_changes.rolling(window=20, min_periods=20).sum()
    
    medium_term_efficiency = close_ratio_20d / sum_abs_changes_20d
    medium_term_efficiency = medium_term_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Efficiency Regime
    efficiency_regime = short_term_efficiency / medium_term_efficiency
    efficiency_regime = efficiency_regime.replace([np.inf, -np.inf], np.nan)
    
    # Fractal Liquidity Microstructure
    # Volume Persistence
    volume_increase = (data['volume'] > data['volume'].shift(1)).astype(int)
    volume_persistence = volume_increase.rolling(window=5, min_periods=1).apply(
        lambda x: x[::-1].cumprod()[::-1].sum(), raw=False
    )
    
    # Amount Efficiency
    avg_trade_price = data['amount'] / data['volume']
    avg_trade_price = avg_trade_price.replace([np.inf, -np.inf], np.nan)
    amount_efficiency = (data['high'] - data['low']) / avg_trade_price
    amount_efficiency = amount_efficiency.replace([np.inf, -np.inf], np.nan)
    
    # Price Impact
    price_impact = abs(data['close'] - data['close'].shift(1)) / data['amount']
    price_impact = price_impact.replace([np.inf, -np.inf], np.nan)
    
    # Alpha Synthesis
    # Core Factor
    core_factor = bull_bear_pressure * intraday_pressure * short_term_efficiency
    
    # Regime Adjusted
    regime_adjusted_factor = core_factor * efficiency_regime
    
    # Volume Confirmed
    volume_confirmed_factor = regime_adjusted_factor * volume_persistence
    
    # Amount Validated
    amount_validated_factor = volume_confirmed_factor * amount_efficiency
    
    # Final Alpha
    sign_efficiency_regime = np.sign(efficiency_regime)
    final_alpha = (amount_validated_factor / (1 + price_impact)) * (1 + abs(gap_momentum)) * sign_efficiency_regime
    
    # Replace any remaining inf/-inf with NaN
    final_alpha = final_alpha.replace([np.inf, -np.inf], np.nan)
    
    return final_alpha
