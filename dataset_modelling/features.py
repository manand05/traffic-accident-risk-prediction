import numpy as np

def engineer_features(df, zone_stats=None):
    df_eng = df.copy()
    
    # interaction features
    df_eng['Wind_x_Precip'] = df_eng['Wind_Speed(mph)'] * df_eng['Precipitation(in)']
    df_eng['Hour_x_Weekend'] = df_eng['Hour'] * df_eng['Weekend']
    df_eng['BadWeather'] = (df_eng['Weather_Rain/Drizzle'] + df_eng['Weather_Stormy'] + df_eng['Weather_Visibility Issues']).clip(0, 1)
    df_eng['BadWeather_x_Hour'] = df_eng['BadWeather'] * df_eng['Hour']
    
    # non-linear relationships
    df_eng['Hour_squared'] = df_eng['Hour'] ** 2
    df_eng['WindSpeed_squared'] = df_eng['Wind_Speed(mph)'] ** 2
    df_eng['Precip_squared'] = df_eng['Precipitation(in)'] ** 2
    
    # cyclic encoding that tells model that hour 23 and hour 0 are close together (0 and 23 aren't 23 units apart, but 1 unit apart which is mapped out on a circle)
    df_eng['Hour_sin'] = np.sin(2 * np.pi * df_eng['Hour'] / 24)
    df_eng['Hour_cos'] = np.cos(2 * np.pi * df_eng['Hour'] / 24)
    df_eng['Month_sin'] = np.sin(2 * np.pi * df_eng['Month'] / 12)
    df_eng['Month_cos'] = np.cos(2 * np.pi * df_eng['Month'] / 12)
    df_eng['DayOfWeek_sin'] = np.sin(2 * np.pi * df_eng['Day_of_Week'] / 7)
    df_eng['DayOfWeek_cos'] = np.cos(2 * np.pi * df_eng['Day_of_Week'] / 7)
    
    # zone stats: compute from training only, reuse for val/test
    if zone_stats is None:
        # only called when processing training data
        zone_stats = (df.groupby('Zone_Int_ID')['Accident_Count'].agg(['mean', 'std', 'max']).reset_index())
        zone_stats.columns = ['Zone_Int_ID', 'Zone_Mean', 'Zone_Std', 'Zone_Max']
        zone_stats['Zone_Std'] = zone_stats['Zone_Std'].fillna(0)
    
    df_eng = df_eng.merge(zone_stats, on='Zone_Int_ID', how='left')
    
    return df_eng, zone_stats

# original features
features = [
    'Hour', 'Day_of_Week', 'Month', 'Weekend', 'Holiday', 'Year', 'Zone_Int_ID', 'Amenity', 'Crossing', 'Give_Way', 'Junction',
    'Railway', 'Station', 'Stop', 'Traffic_Signal', 'City_Houston', 'City_Miami', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)',
    'Visibility(mi)', 'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Clear', 'Weather_Cloudy', 'Weather_Dust/Windy',
    'Weather_Rain/Drizzle', 'Weather_Snow/Ice', 'Weather_Stormy', 'Weather_Visibility Issues'
    ]

engineered_features = [
    'Wind_x_Precip', 'Hour_x_Weekend', 'BadWeather', 'BadWeather_x_Hour', 'Hour_squared', 'WindSpeed_squared', 'Precip_squared',
    'Hour_sin', 'Hour_cos', 'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos', 'Zone_Mean', 'Zone_Std', 'Zone_Max'
    ]

regressor_features = features + engineered_features