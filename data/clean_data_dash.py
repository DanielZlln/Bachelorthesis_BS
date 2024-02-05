import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta, date
from pathlib import Path
import os

def get_src_directory(script_directory):
    src_directory = script_directory.parent / 'src'
    return src_directory

def load_data():
    script_directory = Path(__file__).resolve().parent
    src_directory = get_src_directory(script_directory)
    file_path = src_directory / 'Zaehlstelle_Neutor_2020_Stundenauswertung.xlsx'
    neutor_xlsx = pd.read_excel(file_path, 
                                sheet_name=None, 
                                skiprows=2,
                                skipfooter=1,
                                engine='openpyxl')

    df_neutor = pd.concat(neutor_xlsx.values(), ignore_index=True)
    
    #df_neutor['Datum'] = datetime.strftime(df_neutor['Zeit'], '%Y-%m-%d')
    df_neutor['Datum'] = pd.to_datetime(df_neutor['Zeit'], format='%Y-%m-%d').dt.date.astype(str)
    df_neutor['Zeit'] = pd.to_datetime(df_neutor['Zeit']).dt.time
    
    df_neutor = df_neutor.drop(columns='Unnamed: 0')
    
    return df_neutor
    
def neutor_weekday():
    
    df_neutor = load_data()
    
    # Variable fuer Neutor und mean
    
    df_neutor_weekday = df_neutor.groupby('Datum', dropna=False).agg({'Neutor': 'sum', 
                                                                      'Neutor FR stadteinwärts': 'sum',
                                                                      'Neutor FR stadtauswärts': 'sum'})
    df_neutor_weekday.index = pd.to_datetime(df_neutor_weekday.index)
    df_neutor_weekday['Wochentag'] = df_neutor_weekday.index.day_name()
    
    df_neutor_weekday = df_neutor_weekday.reset_index()
    
    return df_neutor_weekday

def neutor_last_week():
    df_neutor_weekday = neutor_weekday()
    
    to_day = date.today()
    new_to_day = to_day + relativedelta(years=-3)
    last_week = date.today() - timedelta(days=7)
    new_date = last_week + relativedelta(years=-3)

    new_date = pd.to_datetime(new_date)
    last_week = pd.to_datetime(last_week)
    new_to_day = pd.to_datetime(new_to_day)


    df_neutor_last_week = df_neutor_weekday.loc[
        (df_neutor_weekday['Datum'] >= new_date) &
        (df_neutor_weekday['Datum'] < last_week) &
        (df_neutor_weekday['Datum'] < new_to_day)
    ]

    return df_neutor_last_week

if __name__ == "__main__":
    df_neutor = load_data()
    df_neutor_weekday = neutor_weekday()
    df_neutor_last_week = neutor_last_week()