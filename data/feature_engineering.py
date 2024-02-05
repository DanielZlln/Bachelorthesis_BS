import pandas as pd
import numpy as np

# Hilfsfunktionen
def ist_berufsverkehr(uhrzeit):
    berufsverkehr_zeiten = ['06:00:00', '07:00:00', '08:00:00', '16:00:00', '17:00:00', '18:00:00']
    return uhrzeit.strftime('%H:%M:%S') in berufsverkehr_zeiten

def get_jahreszeit(datum):
    monat = datum.month
    if monat in [12, 1, 2]:
        return 'Winter'
    elif monat in [3, 4, 5]:
        return 'Frühling'
    elif monat in [6, 7, 8]:
        return 'Sommer'
    else:
        return 'Herbst'
    
def get_wochentag(df):
    df['Datum'] = pd.to_datetime(df['Datum'])
    df['Wochentag'] = df['Datum'].dt.weekday

    # Mapping der Wochentage zu ihren Namen
    wochentage = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
    df['Wochentag'] = df['Wochentag'].map(lambda day: wochentage[day])
    
    return df

# Vorverarbeitungsfunktionen
def preprocess_dataframe(df, shift_n):
    df = pd.DataFrame(df)
    
    # Wetter entfernen
    #columns_to_drop = ['Wetter', 'Temperatur (°C)', 'Luftfeuchtigkeit (%)', 'Regen (mm)', 'Wind (km/h)']
    #df.drop(columns_to_drop, axis=1, inplace=True)

    # Verkehr von vor einer Woche
    df['Verkehr_vor_einer_Woche'] = df['Neutor (gesamt)'].shift(shift_n)
    df['Verkehr_vor_einer_Woche'].fillna(df['Neutor (gesamt)'], inplace=True)
    df['Verkehrs_Differenz'] = df['Neutor (gesamt)'] - df['Verkehr_vor_einer_Woche']

    # Daten für Neutor FR
    df['Neutor_FR_stadtauswärts_vor_einer_Woche'] = df['Neutor FR stadtauswärts'].shift(shift_n)
    df['Neutor_FR_stadteinwärts_vor_einer_Woche'] = df['Neutor FR stadteinwärts'].shift(shift_n)
    df['Neutor_FR_stadtauswärts_vor_einer_Woche'].fillna(df['Neutor FR stadtauswärts'], inplace=True)
    df['Neutor_FR_stadteinwärts_vor_einer_Woche'].fillna(df['Neutor FR stadteinwärts'], inplace=True)
    df['Neutor_FR_stadtauswärts_Differenz'] = df['Neutor FR stadtauswärts'] - df['Neutor_FR_stadtauswärts_vor_einer_Woche']
    df['Neutor_FR_stadteinwärts_Differenz'] = df['Neutor FR stadteinwärts'] - df['Neutor_FR_stadteinwärts_vor_einer_Woche']
    df.drop(['Neutor FR stadtauswärts', 'Neutor FR stadteinwärts'], axis=1, inplace=True)
    
    df_shifted = df.copy()
    df_shifted['Datum_shifted'] = df_shifted['Datum'] + pd.DateOffset(weeks=1)

    # Bestimmen Sie für jeden Tag in df_shifted die Top-4-Verkehrswerte
    df_shifted['Top4_letzte_Woche'] = df_shifted.groupby('Datum')['Neutor (gesamt)'].rank(method='max', ascending=False) <= 4

    # Merge df_o_wetter und df_shifted anhand von Datum und Zeit, um die Top-4-Markierungen zu erhalten
    df = pd.merge(df, df_shifted[['Datum_shifted', 'Zeit', 'Top4_letzte_Woche']], left_on=['Datum', 'Zeit'], right_on=['Datum_shifted', 'Zeit'], how='left')

    # Füllen Sie fehlende Werte in der Top4_letzte_Woche-Spalte
    df['Top4_letzte_Woche'].fillna(False, inplace=True)

    # Löschen Sie die nicht mehr benötigte Datum_shifted-Spalte
    df.drop('Datum_shifted', axis=1, inplace=True)

    # Berufsverkehr
    df['Berufsverkehr'] = df['Zeit'].apply(lambda x: ist_berufsverkehr(x))

    # Jahreszeit
    df['Jahreszeit'] = df['Datum'].apply(get_jahreszeit)

    return df

