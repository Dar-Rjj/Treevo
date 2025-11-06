import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    for i in range(10, len(df)):
        # Current data
        current = df.iloc[i]
        prev_data = {f't-{j}': df.iloc[i-j] for j in range(1, 11)}
        
        # Asymmetric Price Dynamics
        gap_fill_efficiency = ((current['high'] - max(current['open'], current['close'])) / 
                              (current['high'] - current['low']) - 
                              (min(current['open'], current['close']) - current['low']) / 
                              (current['high'] - current['low']))
        
        price_rejection_strength = (abs(current['close'] - current['open']) / 
                                   (current['high'] - current['low']) * 
                                   np.sign(current['close'] - current['open']) * 
                                   np.sign(prev_data['t-1']['close'] - prev_data['t-1']['open']))
        
        range_asymmetry = ((current['high'] - current['close']) / 
                          (current['close'] - current['low'])) if current['close'] > current['low'] else 0
        
        # Multi-Scale Volume Dynamics
        volume_concentration = (current['volume'] / 
                               (prev_data['t-1']['volume'] + prev_data['t-2']['volume'] + prev_data['t-3']['volume']))
        
        volume_efficiency = ((current['close'] - current['open']) * current['volume'] / 
                            abs(current['open'] - prev_data['t-1']['close']))
        
        trade_size_volatility = (abs(current['amount']/current['volume'] - prev_data['t-1']['amount']/prev_data['t-1']['volume']) * 
                                current['volume'] / prev_data['t-1']['volume'])
        
        # Momentum-Volume Resonance
        momentum_divergence = ((current['close']/prev_data['t-1']['close'] - 1) - 
                              (current['close']/prev_data['t-3']['close'] - 1) - 
                              (current['close']/prev_data['t-8']['close'] - 1))
        
        upside = (current['close'] - current['low']) * current['volume'] if current['close'] > prev_data['t-1']['close'] else 0
        downside = (current['high'] - current['close']) * current['volume'] if current['close'] < prev_data['t-1']['close'] else 0
        
        volume_momentum = ((current['close'] - prev_data['t-1']['close']) * current['volume'] / 
                          (current['high'] - current['low'])) if current['close'] > prev_data['t-1']['close'] else 0
        
        # Microstructure Regime Detection
        market_impact_efficiency = (abs(current['close'] - current['open']) * current['volume'] / current['amount'] * 
                                   (current['high'] - current['low']) / current['close'])
        
        liquidity_absorption = ((current['close'] - current['open']) / (current['high'] - current['low']) * 
                               current['volume'] / prev_data['t-1']['volume'] * 
                               current['amount'] / prev_data['t-1']['amount'])
        
        # Regime Classification
        bull_regime = (current['close'] > prev_data['t-1']['close'] and 
                      current['volume'] > prev_data['t-1']['volume'])
        bear_regime = (current['close'] < prev_data['t-1']['close'] and 
                      current['volume'] < prev_data['t-1']['volume'])
        mixed_regime = not (bull_regime or bear_regime)
        
        # Cross-Timeframe Integration
        short_term_pressure = ((current['close'] - current['open']) / (current['high'] - current['low']) * 
                              current['volume'] / prev_data['t-1']['volume'])
        
        medium_term_flow = ((current['close'] - prev_data['t-2']['close']) / (current['high'] - current['low']) * 
                           current['amount'] / current['volume'])
        
        # Calculate rolling ranges for medium and long term
        high_window_2 = [prev_data['t-2']['high'], prev_data['t-1']['high'], current['high']]
        low_window_2 = [prev_data['t-2']['low'], prev_data['t-1']['low'], current['low']]
        avg_range_medium = (max(high_window_2) - min(low_window_2)) / 3
        
        high_window_10 = [prev_data[f't-{j}']['high'] for j in range(1, 10)] + [current['high']]
        low_window_10 = [prev_data[f't-{j}']['low'] for j in range(1, 10)] + [current['low']]
        max_high_10 = max(high_window_10)
        min_low_10 = min(low_window_10)
        
        momentum_asymmetry_short = ((current['close'] - prev_data['t-1']['close']) / (current['high'] - current['low']) * 
                                   current['volume'] / prev_data['t-1']['volume'])
        
        momentum_asymmetry_medium = ((current['close'] - prev_data['t-3']['close']) / avg_range_medium) if avg_range_medium > 0 else 0
        
        momentum_asymmetry_long = ((current['close'] - prev_data['t-10']['close']) / 
                                  (max_high_10 - min_low_10)) if (max_high_10 - min_low_10) > 0 else 0
        
        # Core Factor Construction
        asymmetric_momentum = (gap_fill_efficiency * (upside - downside) * 
                              current['volume'] / prev_data['t-1']['volume'])
        
        volume_resonance = ((volume_momentum - volume_efficiency) * 
                           volume_momentum / (abs(volume_efficiency) + 1e-6))
        
        microstructure_enhancement = (range_asymmetry * upside / (downside + 1e-6) * 
                                     (1 + abs(current['close'] - current['open']) / (current['high'] - current['low'])))
        
        # Regime-Enhanced Signals
        bull_alpha = ((asymmetric_momentum + volume_resonance) * 
                     (current['high'] - current['open']) / (current['open'] - current['low'])) if current['open'] > current['low'] else 0
        
        bear_alpha = ((asymmetric_momentum + volume_resonance) * 
                     (current['close'] - current['low']) / (current['high'] - current['close'])) if current['high'] > current['close'] else 0
        
        mixed_alpha = ((asymmetric_momentum + volume_resonance) * 
                      (current['close'] - current['open']) / (current['high'] - current['low']))
        
        # Adaptive Components
        volatility_component = ((current['close']/prev_data['t-1']['close'] - 1) / (current['high'] - current['low']) * 
                               current['volume'] / current['amount'])
        
        price_level_component = ((current['close'] - current['open']) / current['close'] * 
                                current['volume'] / prev_data['t-1']['volume'])
        
        # Regime-Weighted Integration
        bull_weighted = bull_alpha * volatility_component * price_level_component if bull_regime else 0
        bear_weighted = bear_alpha * volatility_component * price_level_component if bear_regime else 0
        mixed_weighted = mixed_alpha * volatility_component * price_level_component if mixed_regime else 0
        
        # Final Alpha
        final_alpha = bull_weighted + bear_weighted + mixed_weighted
        
        alpha.iloc[i] = final_alpha
    
    return alpha
