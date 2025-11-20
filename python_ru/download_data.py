# -*- coding: utf-8 -*-
"""
Скрипт для загрузки данных из Google Drive

Этот скрипт загружает все необходимые файлы данных для примеров из книги
"Practical Statistics for Data Scientists: 50 Essential Concepts"
"""
import os
import gdown

# Определение путей
PSDS_PATH = os.path.join(os.path.expanduser("~"), "statistics-for-data-scientists")
DATA_PATH = os.path.join(PSDS_PATH, "data")

# Создание директории для данных, если она не существует
os.makedirs(DATA_PATH, exist_ok=True)

# Словарь с ID файлов Google Drive и именами файлов
data_files = {
    "0B98qpkK5EJembFc5RmVKVVJPdGc": "state.csv",
    "0B98qpkK5EJemcmZYX2VhMHBXelE": "dfw_airline.csv",
    "0B98qpkK5EJemMzZYZHZJaF9va0U": "airline_stats.csv",
    "0B98qpkK5EJemV2htZWdhVFRMNlU": "sp500_px.csv",
    "0B98qpkK5EJemY0U0N1N6a21lUzA": "sp500_sym.csv",
    "0B98qpkK5EJemck5VWkszN3F3RGM": "kc_tax.csv",
    "0B98qpkK5EJemRXpfa2lONlFRSms": "lc_loans.csv",
    "1J96vAqyh92VIeh7kBFm1NBfZcvx8wp2s": "full_train_set.csv",
    "0B98qpkK5EJemd0JnQUtjb051dTA": "loan200.csv",
    "0B98qpkK5EJemQXYtYmJUVkdsN1U": "loan3000.csv",
    "0B98qpkK5EJemZzdoQ2I3SWlBYzg": "loan_data.csv",
    "0B98qpkK5EJemRXVld0NSbWhYNVU": "loans_income.csv",
    "0B98qpkK5EJemOC0xMHBTTEowYzg": "web_page_data.csv",
    "0B98qpkK5EJemOFdZM1JsaEF0Mnc": "four_sessions.csv",
    "0B98qpkK5EJemVHB0ZzdtUG9SeTg": "click_rates.csv",
    "0B98qpkK5EJemZTJnUDd5Ri1vRDA": "imanishi_data.csv",
    "0B98qpkK5EJemb25YYUFJZnZVSnM": "LungDisease.csv",
    "0B98qpkK5EJemWGRWOEhYN1RabVk": "County_Zhvi_AllHomes.csv",
    "0B98qpkK5EJemVTRRN0dLakxwTmM": "house_sales.csv",
}

print("Начинаю загрузку данных из Google Drive...")
print(f"Файлы будут сохранены в: {DATA_PATH}\n")

for file_id, filename in data_files.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    output_path = os.path.join(DATA_PATH, filename)
    
    print(f"Загрузка {filename}...")
    try:
        gdown.download(url, output_path, quiet=False)
        print(f"✓ {filename} успешно загружен\n")
    except Exception as e:
        print(f"✗ Ошибка при загрузке {filename}: {e}\n")

print("Загрузка завершена!")

