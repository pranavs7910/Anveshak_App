import pandas as pd
base_lat = 12.990123
base_long =  80.223452 

df=pd.read_csv('dgps_rover_data.csv')


lat_error = df['Base Latitude (Measured)'] - base_lat
lon_error = df['Base Longitude (Measured)'] - base_long

df['Rover Latitude (Corrected)'] = df['Rover Latitude (Measured)'] - lat_error
df['Rover Longitude (Corrected)'] = df['Rover Longitude (Measured)'] - lon_error

df.to_csv('corrected_gnss_data.csv', index=False)

print("Calculation complete. Corrected data saved to 'corrected_gnss_data.csv'.")
print(df[['Rover Latitude (Corrected)', 'Rover Longitude (Corrected)']].head())