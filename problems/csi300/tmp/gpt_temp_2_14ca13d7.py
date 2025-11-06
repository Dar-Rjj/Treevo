import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    for i in range(len(df)):
        if i < 7:  # Need at least 7 days for calculations
            result.iloc[i] = 0
            continue
            
        current_data = df.iloc[i]
        
        # Multi-Scale Momentum-Volatility Components
        # Micro Momentum-Volatility
        micro_mv = ((current_data['close'] - current_data['open']) / 
                   (current_data['high'] - current_data['low'] + 1e-8)) * \
                  (current_data['high'] - current_data['low'])
        
        # Short Momentum-Volatility (2-day window)
        if i >= 2:
            window_2d = df.iloc[i-2:i+1]
            short_mv = ((current_data['close'] / df.iloc[i-2]['close'] - 1) / 
                       (window_2d['high'].max() - window_2d['low'].min() + 1e-8)) * \
                      current_data['close'] * \
                      (window_2d['high'].max() - window_2d['low'].min())
        else:
            short_mv = 0
            
        # Medium Momentum-Volatility (7-day window)
        window_7d = df.iloc[i-7:i+1]
        medium_mv = ((current_data['close'] / df.iloc[i-7]['close'] - 1) / 
                    (window_7d['high'].max() - window_7d['low'].min() + 1e-8)) * \
                   current_data['close'] * \
                   (window_7d['high'].max() - window_7d['low'].min())
        
        # Flow-Volatility Asymmetry Dynamics
        # Directional Flow Components
        upward_flow = ((current_data['close'] - current_data['low']) / 
                      (current_data['high'] - current_data['low'] + 1e-8)) * current_data['volume']
        downward_flow = ((current_data['high'] - current_data['close']) / 
                        (current_data['high'] - current_data['low'] + 1e-8)) * current_data['volume']
        net_flow = upward_flow - downward_flow
        
        # Flow-Volatility Skewness
        flow_vol_skew = (net_flow * (current_data['high'] - current_data['low'])) / \
                        (abs(net_flow) * (current_data['high'] - current_data['low']) + 1)
        
        # Flow Persistence (3-day window)
        if i >= 3:
            flow_window = df.iloc[i-3:i+1]
            net_flows = []
            for j in range(i-3, i+1):
                day_data = df.iloc[j]
                up_flow = ((day_data['close'] - day_data['low']) / 
                          (day_data['high'] - day_data['low'] + 1e-8)) * day_data['volume']
                down_flow = ((day_data['high'] - day_data['close']) / 
                            (day_data['high'] - day_data['low'] + 1e-8)) * day_data['volume']
                net_flows.append(up_flow - down_flow)
            
            flow_persistence = sum(1 for nf in net_flows if nf > 0) - \
                             sum(1 for nf in net_flows if nf < 0)
        else:
            flow_persistence = 0
        
        # Volume-Momentum-Volatility Dynamics
        if i >= 1:
            # Volume-Momentum Divergence
            vol_mom_div = ((current_data['volume'] / df.iloc[i-1]['volume'] - 1) - 
                          (micro_mv / ((df.iloc[i-1]['close'] - df.iloc[i-1]['open']) / 
                           (df.iloc[i-1]['high'] - df.iloc[i-1]['low'] + 1e-8) * 
                           (df.iloc[i-1]['high'] - df.iloc[i-1]['low']) + 1e-8) - 1))
        else:
            vol_mom_div = 0
        
        # Volume Confirmation (5-day window)
        if i >= 4:
            vol_window = df.iloc[i-4:i+1]
            avg_volume = vol_window['volume'].mean()
            vol_confirmation = (1 if (current_data['volume'] > avg_volume and micro_mv > 0) else 0) - \
                             (1 if (current_data['volume'] < avg_volume and micro_mv < 0) else 0)
        else:
            vol_confirmation = 0
        
        # Trade Size Momentum-Volatility
        if i >= 4:
            trade_size_mv = ((current_data['amount'] / current_data['volume']) / 
                            (df.iloc[i-4]['amount'] / df.iloc[i-4]['volume'] + 1e-8)) * \
                           (current_data['high'] - current_data['low'])
        else:
            trade_size_mv = 1
        
        # Fractal Momentum-Volatility Regime Detection
        # Momentum-Volatility Regime
        if i >= 7:
            mv_window = df.iloc[i-7:i+1]
            medium_mvs = []
            for j in range(i-7, i+1):
                day_data = df.iloc[j]
                if j >= 7:
                    m_window = df.iloc[j-7:j+1]
                    mmv = ((day_data['close'] / df.iloc[j-7]['close'] - 1) / 
                          (m_window['high'].max() - m_window['low'].min() + 1e-8)) * \
                         day_data['close'] * \
                         (m_window['high'].max() - m_window['low'].min())
                    medium_mvs.append(mmv)
            avg_medium_mv = np.mean(medium_mvs) if medium_mvs else 0
            momentum_vol_regime = 2 if medium_mv > avg_medium_mv else 0.5
        else:
            momentum_vol_regime = 1
        
        # Flow-Volatility Regime
        flow_vol_regime = 1.5 if abs(flow_vol_skew) > 0.3 else 1
        
        # Transition Signal
        transition_signal = 1.2 if (micro_mv / (medium_mv + 1e-8)) > 1.2 else 1
        
        # Microstructure-Anchored Momentum Signals
        # Opening Anchor Momentum
        opening_anchor = ((current_data['open'] - (df.iloc[i-1]['high'] + df.iloc[i-1]['low'])/2) / 
                         (df.iloc[i-1]['high'] - df.iloc[i-1]['low'] + 1e-8)) * micro_mv
        
        # Intraday Anchor Efficiency
        intraday_anchor = ((current_data['close'] - current_data['open']) / 
                          (current_data['high'] - current_data['low'] + 1e-8)) * current_data['volume']
        
        # Closing Anchor Pressure
        closing_anchor = (abs(current_data['close'] - (current_data['high'] + current_data['low'])/2) / 
                         (current_data['high'] - current_data['low'] + 1e-8)) * medium_mv
        
        # Dynamic Momentum-Volatility Integration
        if i >= 1:
            prev_micro_mv = ((df.iloc[i-1]['close'] - df.iloc[i-1]['open']) / 
                            (df.iloc[i-1]['high'] - df.iloc[i-1]['low'] + 1e-8)) * \
                           (df.iloc[i-1]['high'] - df.iloc[i-1]['low'])
            
            prev_up_flow = ((df.iloc[i-1]['close'] - df.iloc[i-1]['low']) / 
                           (df.iloc[i-1]['high'] - df.iloc[i-1]['low'] + 1e-8)) * df.iloc[i-1]['volume']
            prev_down_flow = ((df.iloc[i-1]['high'] - df.iloc[i-1]['close']) / 
                             (df.iloc[i-1]['high'] - df.iloc[i-1]['low'] + 1e-8)) * df.iloc[i-1]['volume']
            prev_net_flow = prev_up_flow - prev_down_flow
            
            # Momentum-Volatility-Flow Alignment
            mv_flow_align = np.sign(micro_mv - prev_micro_mv) * np.sign(net_flow) * \
                           np.sign(current_data['high'] - current_data['low'])
            
            # Fractal-Flow-Volatility Consistency
            fractal_flow_vol = (np.log(abs(medium_mv) + 1) / 
                               np.log(abs(current_data['high'] - current_data['low']) + 1)) * flow_persistence
            
            # Momentum-Volatility-Flow Divergence
            mv_flow_div = ((micro_mv / (prev_micro_mv + 1e-8) - 1) - 
                          (net_flow / (prev_net_flow + 1e-8) - 1) - 
                          ((current_data['high'] - current_data['low']) / 
                           (df.iloc[i-1]['high'] - df.iloc[i-1]['low'] + 1e-8) - 1))
        else:
            mv_flow_align = 1
            fractal_flow_vol = 1
            mv_flow_div = 0
        
        # Volume-Confirmed Framework
        if i >= 4:
            # Volume-Momentum-Volatility Alignment
            vol_mom_vol_align = np.sign(current_data['volume'] / df.iloc[i-4]['volume'] - 1) * \
                               np.sign(medium_mv / ((df.iloc[i-4]['close'] / df.iloc[i-11]['close'] - 1) / 
                                       (df.iloc[i-11:i-3]['high'].max() - df.iloc[i-11:i-3]['low'].min() + 1e-8) * 
                                       df.iloc[i-4]['close'] * 
                                       (df.iloc[i-11:i-3]['high'].max() - df.iloc[i-11:i-3]['low'].min()) + 1e-8) - 1) * \
                               np.sign((current_data['high'] - current_data['low']) / 
                                      (df.iloc[i-4]['high'] - df.iloc[i-4]['low'] + 1e-8) - 1)
            
            # Volume Confirmation Score
            vol_window_5d = df.iloc[i-4:i+1]
            avg_vol_5d = vol_window_5d['volume'].mean()
            vol_conf_score = (1 if (current_data['volume'] > avg_vol_5d and micro_mv > 0 and 
                                  (current_data['high'] - current_data['low']) > 1) else 0) - \
                           (1 if (current_data['volume'] < avg_vol_5d and micro_mv < 0 and 
                                (current_data['high'] - current_data['low']) < 1) else 0)
        else:
            vol_mom_vol_align = 1
            vol_conf_score = 0
        
        # Size-Weighted Momentum-Volatility
        size_weighted_mv = micro_mv * (current_data['amount'] / current_data['volume']) * \
                          (current_data['high'] - current_data['low'])
        
        # Composite Alpha Construction
        # Core Momentum-Volatility Flow
        core_mv_flow = net_flow * flow_vol_skew * (current_data['high'] - current_data['low'])
        
        # Volume-Enhanced Core
        vol_enhanced_core = core_mv_flow * vol_confirmation * trade_size_mv
        
        # Microstructure Adjustment
        microstructure_adj = vol_enhanced_core * opening_anchor * intraday_anchor * closing_anchor
        
        # Regime Refinement
        regime_refinement = microstructure_adj * momentum_vol_regime * flow_persistence
        
        # Alignment Enhancement
        alignment_enhancement = regime_refinement * (1 + abs(mv_flow_align))
        
        # Final Alpha Output
        final_alpha = alignment_enhancement * fractal_flow_vol * vol_mom_vol_align
        
        result.iloc[i] = final_alpha
    
    return result
