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
                time, prev_j1, prev_j, consommation = parts
                data.append([current_date, time, int(prev_j1), int(prev_j), int(consommation)])
        
        df = pd.DataFrame(data, columns=['Date', 'Time', 'PrévisionJ-1', 'PrévisionJ', 'Consommation'])
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M')
        df = df[['DateTime', 'PrévisionJ-1', 'PrévisionJ', 'Consommation']]
        
        return df
    except Exception as e:
        print(f"Error in parse_energy_data: {e}")
        return pd.DataFrame()