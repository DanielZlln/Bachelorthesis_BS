from datetime import datetime
from pathlib import Path
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
from prophet import Prophet


def get_src_directory(script_directory):
    src_directory = script_directory.parent / "src"
    return src_directory


def clean_neutor_excel(file_name):
    notebook_directory = os.getcwd()
    script_directory = Path(notebook_directory)
    src_directory = get_src_directory(script_directory)
    file_path = src_directory / file_name
    neutor_xlsx = pd.read_excel(
        file_path, sheet_name=None, skiprows=2, skipfooter=1, engine="openpyxl"
    )
    df_neutor = pd.concat(neutor_xlsx.values(), ignore_index=True)
    new_column_names = {
        "Neutor": "Neutor (gesamt)",
        "FR stadteinwärts": "Neutor FR stadteinwärts",
        "FR stadtauswärts": "Neutor FR stadtauswärts",
        "Time": "Zeit",
    }
    df_neutor = df_neutor.rename(columns=new_column_names)

    if "Gefühlte Temperatur (°C)" in df_neutor.columns:
        df_neutor = df_neutor.drop("Gefühlte Temperatur (°C)", axis=1)

    if "Unnamed: 0" in df_neutor.columns:
        df_neutor = df_neutor.drop(columns="Unnamed: 0", axis=1)

    if "Zeit" in df_neutor.columns:
        success_date = False

        try:
            df_neutor["Datum"] = pd.to_datetime(
                df_neutor["Zeit"], format="%Y-%m-%d"
            ).dt.date.astype(str)
            df_neutor["Zeit"] = pd.to_datetime(df_neutor["Zeit"]).dt.time

            success_date = True
        except ValueError:
            print("Konnte nicht in das Foramt bereinigt werden")

        if not success_date:
            try:
                df_neutor["Zeit"] = df_neutor["Zeit"].astype(str)
                df_neutor[["day", "month", "year", "Uhrzeit"]] = df_neutor[
                    "Zeit"
                ].str.split(" ", expand=True)

                month_to_num = {
                    "Jan.": 1,
                    "Febr.": 2,
                    "Mrz.": 3,
                    "Apr.": 4,
                    "Mai": 5,
                    "Jun.": 6,
                    "Jul.": 7,
                    "Aug.": 8,
                    "Sept.": 9,
                    "Okt.": 10,
                    "Nov.": 11,
                    "Dez.": 12,
                }
                df_neutor.month = df_neutor.month.map(month_to_num)
                df_neutor.day = df_neutor.day.astype(str)
                df_neutor.day = df_neutor.day.str.replace(".", "")

                df_neutor.drop(["Zeit"], axis=True, inplace=True)
                df_neutor["Zeit"] = pd.to_datetime(
                    df_neutor["Uhrzeit"], format="%H:%M", errors="coerce"
                ).dt.time
                df_neutor["Datum"] = pd.to_datetime(df_neutor[["year", "month", "day"]])

                df_neutor.drop(
                    ["day", "month", "year", "Uhrzeit"], axis=True, inplace=True
                )

                success_date = True
            except ValueError:
                print("Konnte nicht in das Foramt bereinigt werden")

        if success_date:
            try:
                df_neutor = df_neutor[
                    [
                        "Datum",
                        "Zeit",
                        "Neutor (gesamt)",
                        "Neutor FR stadteinwärts",
                        "Neutor FR stadtauswärts",
                        "Wetter",
                        "Temperatur (°C)",
                        "Luftfeuchtigkeit (%)",
                        "Regen (mm)",
                        "Wind (km/h)",
                    ]
                ]
            except (ValueError, KeyError):
                print("Falsche Spaltenbezeichnung")

        return df_neutor


def process_all_excels(src_directory):
    files = []
    for root, dirs, files_in_dir in os.walk(src_directory):
        files_in_dir = [
            f for f in files_in_dir if not f[0] == "." and f.endswith(".xlsx")
        ]
        files.extend(files_in_dir)
    data = []
    for i in files:
        df = clean_neutor_excel(i)
        data.append(df)

    df_neutor = pd.concat(data, ignore_index=True)
    df_neutor["Datum"] = pd.to_datetime(df_neutor["Datum"])
    df_neutor.sort_values(["Datum", "Zeit"], ascending=[True, True], inplace=True)
    df_neutor['Wochentag'] = df_neutor['Datum'].dt.weekday
    wochentage_deutsch = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
    df_neutor['Wochentag'] = df_neutor['Wochentag'].map(lambda day: wochentage_deutsch[day])
    
    df_neutor.reset_index(drop=True, inplace=True)

    return df_neutor


def merge_corona_intensity(df_neutor):
    lockdown_data = pd.DataFrame(
        data={
            "Start": ["2020-03-22", "2020-11-02", "2021-01-06"],
            "Ende": ["2020-05-04", "2021-01-05", "2021-05-01"],
            "Intensität": [2, 1, 2],
        }
    )
    lockdown_data["Start"] = pd.to_datetime(lockdown_data["Start"])
    lockdown_data["Ende"] = pd.to_datetime(lockdown_data["Ende"])

    lockdown_intensity = []

    for datum in df_neutor["Datum"]:
        intensity = 0
        for _, row in lockdown_data.iterrows():
            if row["Start"] <= datum <= row["Ende"]:
                intensity = row["Intensität"]
                break
        lockdown_intensity.append(intensity)

    df_neutor["Lockdown"] = lockdown_intensity
    return df_neutor


def get_feiertage(df_neutor):
    today = datetime.today().date()
    year = today.year

    url = f"https://www.spiketime.de/feiertagapi/feiertage/2018/{year}"
    response = requests.get(url)

    if response.status_code != 200:
        print("Fehler beim Abrufen der Daten von der API.")
        return df_neutor

    data_feiertage = response.json()
    df_feiertage = pd.json_normalize(data_feiertage)
    df_feiertage = df_feiertage[
        df_feiertage["Feiertag.Laender"].apply(
            lambda x: any(entry["Name"] == "Nordrhein-Westfalen" for entry in x)
        )
    ]
    df_feiertage = df_feiertage.drop("Feiertag.Laender", axis=1)

    df_neutor["Feiertag"] = pd.Series(0, index=df_neutor.index)
    df_neutor["Feiertag"] = np.where(
        df_neutor["Datum"].isin(df_feiertage["Datum"]), 1, 0
    )

    return df_neutor


def get_semesterferien(df_neutor):
    url = "https://www.uni-muenster.de/studium/orga/termine_archiv.html"
    response = requests.get(url)

    soup = BeautifulSoup(response.text, "html.parser")
    ferien_datum = soup.find("tbody", class_="tab4")
    table_rows = ferien_datum.find_all("tr")

    data = []

    for row in table_rows:
        columns = row.find_all("td")
        row_data = [column.get_text(strip=True) for column in columns]
        data.append(row_data)

    df_semesterferien = pd.DataFrame({"ferien": data})

    column_names = [
        "semester",
        "semester_beginn",
        "vorlesungsbeginn",
        "erster_ferientag",
        "letzter_ferientag",
        "vorlesungsende",
        "semesterende",
    ]

    df_semesterferien[column_names] = pd.DataFrame(
        df_semesterferien["ferien"].tolist(), columns=column_names
    )

    date_columns = [
        "semester_beginn",
        "vorlesungsbeginn",
        "erster_ferientag",
        "letzter_ferientag",
        "vorlesungsende",
        "semesterende",
    ]
    for col in date_columns:
        df_semesterferien[col] = pd.to_datetime(
            df_semesterferien[col], format="%d.%m.%Y"
        )

    df_semesterferien.drop("ferien", axis=1, inplace=True)

    data = []

    for index, row in df_semesterferien.iterrows():
        semester = row["semester"]
        vorlesungsbeginn = row["vorlesungsbeginn"]
        vorlesungsende = row["vorlesungsende"]
        erster_ferientag = row["erster_ferientag"]
        letzter_ferientag = row["letzter_ferientag"]

        current_date = vorlesungsbeginn
        while current_date <= vorlesungsende:
            if erster_ferientag <= current_date <= letzter_ferientag:
                current_date += pd.DateOffset(days=1)
                continue
            data.append({"Semester": semester, "Datum": current_date.date()})
            current_date += pd.DateOffset(days=1)

    result_df = pd.DataFrame(data)
    result_df["Datum"] = pd.to_datetime(result_df["Datum"]).dt.date

    df_neutor["Semesterferien"] = 1
    df_neutor.loc[df_neutor["Datum"].isin(result_df["Datum"]), "Semesterferien"] = 0

    return df_neutor


def get_ferien(df_neutor):
    today = datetime.today().date()
    year = today.year
    start_year = 2017
    years = list(range(start_year, year + 1))

    datum_ferien = []

    for i in years:
        url_ferien = f"https://ferien-api.de/api/v1/holidays/NW/{i}"

        response = requests.get(url_ferien)

        if response.status_code != 200:
            print("Fehler beim Abrufen der Daten von der API.")
            continue

        data_ferien = response.json()
        df_ferien = pd.json_normalize(data_ferien)
        df_ferien = df_ferien.drop("slug", axis=1)

        for index, row in df_ferien.iterrows():
            start_date = datetime.strptime(row["start"], "%Y-%m-%d")
            end_date = datetime.strptime(row["end"], "%Y-%m-%d")
            current_date = start_date
            while current_date <= end_date:
                datum_ferien.append({"Datum": current_date.date()})
                current_date += pd.DateOffset(days=1)

    df_neu = pd.DataFrame(datum_ferien)

    df_neutor["Ferien"] = 0
    df_neutor.loc[df_neutor["Datum"].isin(df_neu["Datum"]), "Ferien"] = 1

    return df_neutor


def save_to_pickle(dataframe, output_path, filename):
    output_path.mkdir(exist_ok=True)
    pkl_path = output_path / filename
    dataframe.to_pickle(pkl_path)


def fill_data(df_neutor):
    numerical_columns = df_neutor.select_dtypes(include=["number", "float"]).columns
    columns_over_zero = [
        column for column in numerical_columns if df_neutor[column].min() >= 0
    ]

    for column in df_neutor.columns:
        missing_count = df_neutor[column].isnull().sum()

        if missing_count > 0:
            if missing_count > 100:
                # Füllen mit Prophet
                df_prophet = df_neutor[["Datum", column]].copy()
                df_prophet.columns = ["ds", "y"]
                df_prophet["ds"] = pd.to_datetime(df_prophet["ds"])

                model = Prophet()
                model.fit(df_prophet)

                future = model.make_future_dataframe(periods=0)
                forecast = model.predict(future)

                filled_data_pro = forecast[["ds", "yhat"]]
                filled_data_pro["ds"] = pd.to_datetime(filled_data_pro["ds"])

                df_neutor_filled_pro = pd.merge(
                    df_neutor,
                    filled_data_pro,
                    left_on="Datum",
                    right_on="ds",
                    how="left",
                )

                df_neutor_filled_pro[column] = df_neutor_filled_pro[column].fillna(
                    df_neutor_filled_pro["yhat"]
                )

                df_neutor_filled_pro.drop(columns=["ds", "yhat"], inplace=True)

                df_neutor[column] = df_neutor_filled_pro[column]
            else:
                # Füllen mit Mittelwert
                df_neutor[column] = df_neutor[column].fillna(df_neutor[column].mean())

    for column in columns_over_zero:
        positive_mean = df_neutor[column].loc[df_neutor[column] >= 0].mean()
        negative_indices = df_neutor[df_neutor[column] < 0].index
        df_neutor.loc[negative_indices, column] = positive_mean

    return df_neutor


if __name__ == "__main__":
    src_directory = "/Users/danielzellner/Documents/Studium/Bachelorthesis/src"
    df_neutor = process_all_excels(src_directory)
    df_neutor = get_feiertage(df_neutor)
    df_neutor = get_semesterferien(df_neutor)
    df_neutor = get_ferien(df_neutor)
    df_neutor = merge_corona_intensity(df_neutor)

    new_order = [
        "Datum",
        "Zeit",
        "Wochentag",
        "Neutor (gesamt)",
        "Neutor FR stadteinwärts",
        "Neutor FR stadtauswärts",
        "Wetter",
        "Temperatur (°C)",
        "Luftfeuchtigkeit (%)",
        "Regen (mm)",
        "Wind (km/h)",
        "Feiertag",
        "Semesterferien",
        "Ferien",
        "Lockdown",
    ]

    df_neutor = df_neutor[new_order]
    output_path = Path(src_directory) / "verkehr_data"
    
    save_to_pickle(df_neutor, output_path, "df_neutor.pkl")
    
    df_neutor = fill_data(df_neutor)

    save_to_pickle(df_neutor, output_path, "df_neutor_complete.pkl")
