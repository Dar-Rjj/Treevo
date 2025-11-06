import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize all intermediate columns
    data['Bullish_Micro_Range'] = np.nan
    data['Bearish_Micro_Range'] = np.nan
    data['Fractal_Range_Asymmetry'] = np.nan
    data['Fractal_Midpoint'] = (data['high'] + data['low']) / 2
    data['Volume_Weighted_Oscillation'] = np.nan
    data['Directional_Micro_Oscillation'] = np.nan
    data['Microstructure_Price_Movement'] = data['close'] - data['open']
    data['Trade_Size_Efficiency'] = (data['amount'] / data['volume']) / (data['high'] - data['low'])
    data['Price_Trade_Size_Divergence'] = np.nan
    data['Volume_Fractal_Momentum'] = np.nan
    data['Trade_Size_Momentum'] = np.nan
    data['Cross_Dimensional_Persistence'] = np.nan
    data['Fractal_Regime_Strength'] = np.nan
    data['Trade_Size_Reversal_Strength'] = np.nan
    data['Extreme_Micro_Reversal'] = np.nan
    data['Fractal_Center_Deviation'] = data['close'] - data['Fractal_Midpoint']
    data['Trade_Size_Rejection'] = np.nan
    data['Micro_Rejection_Power'] = np.nan
    data['Trade_Size_Flow_Pressure'] = data['volume'] / (data['high'] - data['low'])
    data['Weighted_Extreme_Micro_Reversal'] = np.nan
    data['Weighted_Midpoint_Rejection'] = np.nan
    data['Trade_Size_Volatility_Ratio'] = np.nan
    data['Volatility_Trade_Size_Signal'] = np.nan
    data['Fractal_Micro_Convergence'] = np.nan
    data['Trade_Size_Volatility_Divergence'] = np.nan
    data['Micro_Regime_Confirmation'] = np.nan
    data['Micro_Filter_Strength'] = np.nan
    data['Rejection_Component'] = np.nan
    data['Volatility_Component'] = np.nan
    data['Trade_Size_Component'] = np.nan
    data['Regime_Component'] = np.nan
    
    # Calculate rolling windows for volatility
    for i in range(len(data)):
        if i >= 20:
            # Fractal True Range with Microstructure
            if data['close'].iloc[i] > data['open'].iloc[i]:
                data.loc[data.index[i], 'Bullish_Micro_Range'] = max(
                    data['high'].iloc[i] - data['low'].iloc[i],
                    abs(data['high'].iloc[i] - data['close'].iloc[i-1])
                )
                data.loc[data.index[i], 'Bearish_Micro_Range'] = max(
                    data['high'].iloc[i] - data['low'].iloc[i],
                    abs(data['low'].iloc[i] - data['close'].iloc[i-1])
                )
            else:
                data.loc[data.index[i], 'Bullish_Micro_Range'] = max(
                    data['high'].iloc[i] - data['low'].iloc[i],
                    abs(data['high'].iloc[i] - data['close'].iloc[i-1])
                )
                data.loc[data.index[i], 'Bearish_Micro_Range'] = max(
                    data['high'].iloc[i] - data['low'].iloc[i],
                    abs(data['low'].iloc[i] - data['close'].iloc[i-1])
                )
            
            # Fractal Range Asymmetry
            if (data['Bullish_Micro_Range'].iloc[i] + data['Bearish_Micro_Range'].iloc[i]) > 0:
                data.loc[data.index[i], 'Fractal_Range_Asymmetry'] = (
                    data['Bullish_Micro_Range'].iloc[i] / 
                    (data['Bullish_Micro_Range'].iloc[i] + data['Bearish_Micro_Range'].iloc[i])
                )
            
            # Volume-Weighted Fractal Oscillation
            if i >= 1:
                vol_ratio = data['volume'].iloc[i] / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 1
                data.loc[data.index[i], 'Volume_Weighted_Oscillation'] = (
                    (data['Fractal_Midpoint'].iloc[i] - data['Fractal_Midpoint'].iloc[i-1]) * vol_ratio
                )
                data.loc[data.index[i], 'Directional_Micro_Oscillation'] = (
                    data['Volume_Weighted_Oscillation'].iloc[i] * np.sign(data['close'].iloc[i] - data['Fractal_Midpoint'].iloc[i])
                )
            
            # Trade Size Fractal Momentum
            micro_vol_5 = sum(data['high'].iloc[i-5:i] - data['low'].iloc[i-5:i]) / 5
            micro_baseline_20 = sum(data['high'].iloc[i-20:i] - data['low'].iloc[i-20:i]) / 20
            if micro_baseline_20 > 0:
                data.loc[data.index[i], 'Trade_Size_Volatility_Ratio'] = micro_vol_5 / micro_baseline_20
            
            # Price-Trade Size Divergence
            data.loc[data.index[i], 'Price_Trade_Size_Divergence'] = (
                data['Microstructure_Price_Movement'].iloc[i] * data['Trade_Size_Efficiency'].iloc[i]
            )
            
            # Volume-Amount Fractal Persistence
            if i >= 1:
                vol_momentum = data['volume'].iloc[i] / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 1
                trade_size_current = data['amount'].iloc[i] / data['volume'].iloc[i] if data['volume'].iloc[i] > 0 else 0
                trade_size_prev = data['amount'].iloc[i-1] / data['volume'].iloc[i-1] if data['volume'].iloc[i-1] > 0 else 0
                
                data.loc[data.index[i], 'Volume_Fractal_Momentum'] = vol_momentum
                if trade_size_prev > 0:
                    data.loc[data.index[i], 'Trade_Size_Momentum'] = trade_size_current / trade_size_prev
                
                data.loc[data.index[i], 'Cross_Dimensional_Persistence'] = (
                    np.sign(data['Volume_Fractal_Momentum'].iloc[i]) * np.sign(data['Trade_Size_Momentum'].iloc[i])
                )
            
            # Fractal Regime Classification
            convergent = 1 if data['Cross_Dimensional_Persistence'].iloc[i] > 0 else 0
            divergent = 1 if data['Cross_Dimensional_Persistence'].iloc[i] < 0 else 0
            data.loc[data.index[i], 'Fractal_Regime_Strength'] = convergent - divergent
            
            # Trade Size Extreme Reversal
            if i >= 2:
                price_dir = np.sign(data['close'].iloc[i-1] - data['close'].iloc[i-2])
                
                if data['close'].iloc[i] > data['open'].iloc[i]:
                    if data['Bullish_Micro_Range'].iloc[i] > 0:
                        data.loc[data.index[i], 'Trade_Size_Reversal_Strength'] = (
                            data['Microstructure_Price_Movement'].iloc[i] / data['Bullish_Micro_Range'].iloc[i]
                        )
                else:
                    if data['Bearish_Micro_Range'].iloc[i] > 0:
                        data.loc[data.index[i], 'Trade_Size_Reversal_Strength'] = (
                            data['Microstructure_Price_Movement'].iloc[i] / data['Bearish_Micro_Range'].iloc[i]
                        )
                
                data.loc[data.index[i], 'Extreme_Micro_Reversal'] = (
                    price_dir * data['Trade_Size_Reversal_Strength'].iloc[i]
                )
            
            # Midpoint Microstructure Rejection
            if i >= 1:
                data.loc[data.index[i], 'Trade_Size_Rejection'] = (
                    data['Directional_Micro_Oscillation'].iloc[i] * np.sign(data['Fractal_Center_Deviation'].iloc[i])
                )
                
                if (data['high'].iloc[i] - data['low'].iloc[i]) > 0:
                    data.loc[data.index[i], 'Micro_Rejection_Power'] = (
                        data['Trade_Size_Rejection'].iloc[i] * 
                        abs(data['Fractal_Center_Deviation'].iloc[i]) / 
                        (data['high'].iloc[i] - data['low'].iloc[i])
                    )
            
            # Cross-Dimensional Weighted Rejection
            data.loc[data.index[i], 'Weighted_Extreme_Micro_Reversal'] = (
                data['Extreme_Micro_Reversal'].iloc[i] * data['Trade_Size_Flow_Pressure'].iloc[i]
            )
            data.loc[data.index[i], 'Weighted_Midpoint_Rejection'] = (
                data['Micro_Rejection_Power'].iloc[i] * data['Trade_Size_Flow_Pressure'].iloc[i]
            )
            
            # Volatility-Trade Size Regime Detection
            high_vol = 1 if data['Trade_Size_Volatility_Ratio'].iloc[i] > 1 else 0
            low_vol = 1 if data['Trade_Size_Volatility_Ratio'].iloc[i] < 0.8 else 0
            data.loc[data.index[i], 'Volatility_Trade_Size_Signal'] = high_vol - low_vol
            
            # Cross-Dimensional Regime Interaction
            data.loc[data.index[i], 'Fractal_Micro_Convergence'] = (
                data['Fractal_Regime_Strength'].iloc[i] * data['Volatility_Trade_Size_Signal'].iloc[i]
            )
            data.loc[data.index[i], 'Trade_Size_Volatility_Divergence'] = (
                data['Trade_Size_Efficiency'].iloc[i] - data['Trade_Size_Volatility_Ratio'].iloc[i]
            )
            data.loc[data.index[i], 'Micro_Regime_Confirmation'] = (
                data['Fractal_Micro_Convergence'].iloc[i] * data['Trade_Size_Volatility_Divergence'].iloc[i]
            )
            
            # Adaptive Microstructure Filtering
            data.loc[data.index[i], 'Micro_Filter_Strength'] = abs(data['Micro_Regime_Confirmation'].iloc[i])
            
            # Component Construction
            data.loc[data.index[i], 'Rejection_Component'] = (
                data['Weighted_Extreme_Micro_Reversal'].iloc[i] * data['Weighted_Midpoint_Rejection'].iloc[i]
            )
            data.loc[data.index[i], 'Volatility_Component'] = (
                data['Fractal_Range_Asymmetry'].iloc[i] * data['Trade_Size_Volatility_Ratio'].iloc[i]
            )
            data.loc[data.index[i], 'Trade_Size_Component'] = (
                data['Trade_Size_Efficiency'].iloc[i] * data['Cross_Dimensional_Persistence'].iloc[i]
            )
            data.loc[data.index[i], 'Regime_Component'] = (
                data['Fractal_Regime_Strength'].iloc[i] * data['Volatility_Trade_Size_Signal'].iloc[i]
            )
    
    # Final Microstructure Alpha Generation
    alpha_values = []
    
    for i in range(len(data)):
        if i >= 20 and not pd.isna(data['Micro_Regime_Confirmation'].iloc[i]):
            # Determine regime for weighting
            high_vol = data['Trade_Size_Volatility_Ratio'].iloc[i] > 1
            convergent = data['Cross_Dimensional_Persistence'].iloc[i] > 0
            
            if high_vol and convergent:
                w_rej, w_vol, w_trade, w_reg = 0.4, 0.3, 0.2, 0.1
            elif not high_vol and convergent:
                w_rej, w_vol, w_trade, w_reg = 0.3, 0.2, 0.4, 0.1
            else:
                w_rej, w_vol, w_trade, w_reg = 0.25, 0.25, 0.25, 0.25
            
            # Weighted Component Sum
            weighted_sum = (
                data['Rejection_Component'].iloc[i] * w_rej +
                data['Volatility_Component'].iloc[i] * w_vol +
                data['Trade_Size_Component'].iloc[i] * w_trade +
                data['Regime_Component'].iloc[i] * w_reg
            )
            
            # Apply filters and enhancements
            filtered_value = weighted_sum * data['Micro_Filter_Strength'].iloc[i]
            
            if not pd.isna(data['Trade_Size_Momentum'].iloc[i]):
                filtered_value *= data['Trade_Size_Momentum'].iloc[i]
            
            # Microstructure Enhancement
            if data['open'].iloc[i] > 0:
                enhancement = (data['Microstructure_Price_Movement'].iloc[i] / data['open'].iloc[i]) * data['Trade_Size_Flow_Pressure'].iloc[i]
                final_alpha = filtered_value + enhancement
            else:
                final_alpha = filtered_value
                
            alpha_values.append(final_alpha)
        else:
            alpha_values.append(np.nan)
    
    return pd.Series(alpha_values, index=data.index)
