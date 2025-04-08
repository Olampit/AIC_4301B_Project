import pandas as pd
import re

def parse_energy_data(file_path):
    #Use of try/catch block to handle errors
    try:

        #We open the xls file and read it line by line
        with open(file_path, 'r', encoding='ISO-8859-1') as file:
            lines = file.readlines()
        
        data = []
        current_date = None

        #Loop on each line of the file
        for line in lines:
            line = line.strip()
            
            #match : allows us to verify if we are on a line that interests us
            match = re.match(r'Journée du (\d{2}/\d{2}/\d{4})', line)
            
            #if we are on a line that looks like "Journée du XX/XX/XXXX" : we extract the date
            if match:
                current_date = match.group(1)
                continue
            
            #if the line starts with "Heures" : we go to the next line
            if line.startswith("Heures"):
                continue
            
            #At this point, line should be formed of 4 fields : "current_date", "prev_j1", "prev_j", "consommation"
            #Each field separated by a tabulation ("\t")
            parts = line.split('\t')
            if len(parts) == 4:

                #Extraction of the values time, prev_j1, prev_j, consommation
                try:
                    time, prev_j1, prev_j, consommation = parts
                    data.append([current_date, time, int(prev_j1), int(prev_j), int(consommation)])
                
                #Error Handling : in case one of fields are empty 
                #(by experience, time and current data are never empty)
                except ValueError:
                    data.append([current_date, time, None, None, None])

            
        #Generation of the dataframe from the extracted values stocked in data
        df = pd.DataFrame(data, columns=['Date', 'Time', 'PrévisionJ-1', 'PrévisionJ', 'Consommation'])
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M')
        df = df[['DateTime', 'PrévisionJ-1', 'PrévisionJ', 'Consommation']]
        df['Date'] = df['DateTime'].dt.date #grouping by date, mandatory line

        

        #Handling of missing data

        #Fill the missing values by the numerical mean of the fields with the same date
        df = df.groupby('Date').apply(lambda group: group.fillna(group.mean(numeric_only=True)))


        #Definition of a threshold for missing data at 50%
        threshold = 0.5 

        #We loop through each unique date in the DataFrame
        for date in df['Date'].unique():
            
            daily_data = df[df['Date'] == date] #Extract the data corresponding to the current date
            missing_ratio = daily_data.isna().mean().max() #Compute the highest missing ratio among numerical columns for the given date

            #If the missing ratio is higher than the treshold
            #Extract the data from the last week
            #And replace the fields previsionj_1, previsionj and consommation with the values of last week
            if missing_ratio > threshold:
                previous_week = pd.to_datetime(date) - pd.Timedelta(days=7) 
                if previous_week.date() in df['Date'].values:
                    df.loc[df['Date'] == date, ['PrévisionJ-1', 'PrévisionJ', 'Consommation']] = \
                        df[df['Date'] == previous_week.date()][['PrévisionJ-1', 'PrévisionJ', 'Consommation']].values

        #Drop the column date of the dataframe (no longer needed)
        df.drop(columns=['Date'], inplace=True)

        #print(f"data head : {df.head()}")

        #Return the completed dataframe
        return df
    
    except Exception as e:
        print(f"Error in parse_energy_data: {e}")
        return pd.DataFrame()
