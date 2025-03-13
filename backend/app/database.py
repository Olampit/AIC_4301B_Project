import pandas as pd
import re

def parse_energy_data(file_path):
    try:
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            lines = file.readlines()
        
        data = []
        current_date = None
        for line in lines:
            line = line.strip()
            
            match = re.match(r'Journée du (\d{2}/\d{2}/\d{4})', line)
            if match:
                current_date = match.group(1)
                continue
            
            if line.startswith("Heures"):
                continue
            
            parts = line.split('\t')
            if len(parts) == 4:
                try:
                    time, prev_j1, prev_j, consommation = parts
                    data.append([current_date, time, int(prev_j1), int(prev_j), int(consommation)])
                except ValueError:
                    data.append([current_date, time, None, None, None])
        
        df = pd.DataFrame(data, columns=['Date', 'Time', 'PrévisionJ-1', 'PrévisionJ', 'Consommation'])
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M')
        df = df[['DateTime', 'PrévisionJ-1', 'PrévisionJ', 'Consommation']]
        df['Date'] = df['DateTime'].dt.date #grouping by date, mandatory line


        #below is missing data handling :

        df = df.groupby('Date').apply(lambda group: group.fillna(group.mean(numeric_only=True)))

        threshold = 0.5 #50% is 0.5
        for date in df['Date'].unique():
            daily_data = df[df['Date'] == date]
            missing_ratio = daily_data.isna().mean().max() 

            if missing_ratio > threshold:
                previous_week = pd.to_datetime(date) - pd.Timedelta(days=7)
                if previous_week.date() in df['Date'].values:
                    df.loc[df['Date'] == date, ['PrévisionJ-1', 'PrévisionJ', 'Consommation']] = \
                        df[df['Date'] == previous_week.date()][['PrévisionJ-1', 'PrévisionJ', 'Consommation']].values

        df.drop(columns=['Date'], inplace=True)

        return df
    except Exception as e:
        print(f"Error in parse_energy_data: {e}")
        return pd.DataFrame()
