import pandas as pd
    import numpy as np

    # Calculate Parabolic SAR
    high = df['high']
    low = df['low']
    close = df['close']
    
    af = 0.02
    af_max = 0.2
    sar = [min(high[:2]) if close[1] > close[0] else max(low[:2])]
    ep = [max(high[:2]) if close[1] > close[0] else min(low[:2])]
    is_long = close[1] > close[0]
    ac_factor = af
    
    for i in range(2, len(close)):
        if is_long:
            if low[i] < sar[-1]:
                is_long = False
                sar.append(ep[-1])
                ep.append(low[i])
                ac_factor = af
            else:
                sar.append(sar[-1] + ac_factor * (ep[-1] - sar[-1]))
                if high[i] > ep[-1]:
                    ep.append(high[i])
                    ac_factor = min(ac_factor + af, af_max)
                else:
                    ep.append(ep[-1])
        else:
            if high[i] > sar[-1]:
                is_long = True
                sar.append(ep[-1])
                ep.append(high[i])
                ac_factor = af
            else:
                sar.append(sar[-1] + ac_factor * (ep[-1] - sar[-1]))
                if low[i] < ep[-1]:
                    ep.append(low[i])
                    ac_factor = min(ac_factor + af, af_max)
                else:
                    ep.append(ep[-1])
    
    df['sar'] = sar
    df['distance_to_sar'] = df['close'] - df['sar']

    # Calculate Accumulation/Distribution Line
    adl = ((close - low) - (high - close)) / (high - low) * df['volume']
    adl_cumulative = adl.cumsum()
    
    # Combine factors
    heuristics_matrix = df['distance_to_sar'] + adl_cumulative
    return heuristics_matrix
