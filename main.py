import sqlite3
import os
import csv
import random
from datetime import datetime
from collections import defaultdict
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import matplotlib.pyplot as plt
import io
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import pandas as pd
from typing import Dict, List, Tuple
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Регистрация шрифтов (укажите правильные пути к файлам)
pdfmetrics.registerFont(TTFont('DejaVuSans', 'DejaVuSans.ttf'))
pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', 'DejaVuSans-Bold.ttf'))
pdfmetrics.registerFontFamily('DejaVuSans', normal='DejaVuSans', bold='DejaVuSans-Bold')


class DatabaseManager:
    """Управление базой данных SQLite с функционалом загрузки и обновления"""

    def __init__(self, db_name='predprof_vit.db'):
        self.db_name = db_name
        self.op_names = ['ПМ', 'ИВТ', 'ИТСС', 'ИБ']
        self.dates = ['01_08', '02_08', '03_08', '04_08']
        self.places = {'ПМ': 40, 'ИВТ': 50, 'ИТСС': 30, 'ИБ': 20}

        # Создание/проверка структуры БД
        self.create_database()

    def create_database(self):
        """Создание структуры БД (16 таблиц) с правильными названиями столбцов"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        for op in self.op_names:
            for date in self.dates:
                table_name = f"{op}_{date}"
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id_applic INTEGER PRIMARY KEY,
                        consent BOOLEAN NOT NULL,
                        priorit INTEGER NOT NULL,
                        physics_it_socre INTEGER NOT NULL,
                        russian_score INTEGER NOT NULL,
                        math_score INTEGER NOT NULL,
                        achivments_score INTEGER NOT NULL,
                        total_score INTEGER NOT NULL
                    )
                ''')

        conn.commit()
        conn.close()
        print(f"✓ База данных '{self.db_name}' готова")

    def get_all_applicants(self, date):
        """
        Получение всех уникальных абитуриентов за указанную дату со всеми их заявлениями.
        """
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        applicants = defaultdict(lambda: {
            'id': None,
            'total_score': 0,
            'math_score': 0,
            'russian_score': 0,
            'physics_it_socre': 0,
            'achivments_score': 0,
            'applications': []
        })

        for op in self.op_names:
            table_name = f"{op}_{date}"
            try:
                cursor.execute(f'''
                    SELECT id_applic, consent, priorit, math_score, russian_score, 
                           physics_it_socre, achivments_score, total_score 
                    FROM {table_name}
                ''')

                for row in cursor.fetchall():
                    app_id = row[0]
                    if applicants[app_id]['id'] is None:
                        applicants[app_id].update({
                            'id': app_id,
                            'math_score': row[3],
                            'russian_score': row[4],
                            'physics_it_socre': row[5],
                            'achivments_score': row[6],
                            'total_score': row[7]
                        })

                    applicants[app_id]['applications'].append({
                        'program': op,
                        'priority': row[2],
                        'agreement': bool(row[1])
                    })
            except sqlite3.OperationalError:
                # Таблица может отсутствовать при пустой БД
                continue

        conn.close()
        return list(applicants.values())

    def get_program_list(self, program, date):
        """Получение списка абитуриентов для конкретной программы и даты"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()
        table_name = f"{program}_{date}"

        try:
            cursor.execute(f'''
                SELECT id_applic, consent, priorit, physics_it_socre, russian_score, 
                       math_score, achivments_score, total_score
                FROM {table_name}
                ORDER BY total_score DESC, id_applic ASC
            ''')

            columns = ['id_applic', 'consent', 'priorit', 'physics_it_socre', 'russian_score',
                       'math_score', 'achivments_score', 'total_score']
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        except sqlite3.OperationalError:
            results = []

        conn.close()
        print(f"Таблица {table_name}: найдено {len(results)} записей")
        return results

    def import_from_csv(self, csv_path: str, date: str) -> Tuple[bool, str, Dict]:
        """
        Загрузка данных из CSV-файла в БД.
        Формат CSV: id_applic,consent,priorit,physics_it_socre,russian_score,math_score,achivments_score,total_score,program

        Возвращает: (успех, сообщение, статистика операций)
        """
        start_time = datetime.now()

        # Проверка существования файла
        if not os.path.exists(csv_path):
            return False, f"Файл не найден: {csv_path}", {}

        # Чтение CSV
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
        except Exception as e:
            return False, f"Ошибка чтения CSV: {e}", {}

        # Валидация структуры (с правильными названиями столбцов)
        required_columns = ['id_applic', 'consent', 'priorit', 'physics_it_socre', 'russian_score',
                            'math_score', 'achivments_score', 'total_score', 'program']
        if not all(col in df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in df.columns]
            return False, f"Неверная структура CSV. Отсутствуют колонки: {missing}", {}

        # Группировка по программам
        programs_data = {op: [] for op in self.op_names}
        for _, row in df.iterrows():
            op = str(row['program']).strip()
            if op in self.op_names:
                programs_data[op].append({
                    'id_applic': int(row['id_applic']),
                    'consent': 1 if str(row['consent']).lower() in ['1', 'true', 'да', 'yes', 'y'] else 0,
                    'priorit': int(row['priorit']),
                    'physics_it_socre': int(row['physics_it_socre']),
                    'russian_score': int(row['russian_score']),
                    'math_score': int(row['math_score']),
                    'achivments_score': int(row['achivments_score']),
                    'total_score': int(row['total_score'])
                })

        # Подключение к БД
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        stats = {
            'deleted': 0,
            'added': 0,
            'updated': 0,
            'total': 0
        }

        # Для каждой программы выполняем операции обновления
        for op in self.op_names:
            table_name = f"{op}_{date}"

            # Получение текущих данных из БД (если таблица существует)
            existing_ids = set()
            try:
                cursor.execute(f"SELECT id_applic FROM {table_name}")
                existing_ids = {row[0] for row in cursor.fetchall()}
            except sqlite3.OperationalError:
                # Таблица не существует - создаем
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id_applic INTEGER PRIMARY KEY,
                        consent BOOLEAN NOT NULL,
                        priorit INTEGER NOT NULL,
                        physics_it_socre INTEGER NOT NULL,
                        russian_score INTEGER NOT NULL,
                        math_score INTEGER NOT NULL,
                        achivments_score INTEGER NOT NULL,
                        total_score INTEGER NOT NULL
                    )
                ''')

            new_ids = {app['id_applic'] for app in programs_data[op]}

            # 1. Удаление записей (5-10% от общего количества)
            ids_to_delete = existing_ids - new_ids
            # Ограничиваем удаление до 10% от общего количества записей
            max_delete = max(1, int(len(existing_ids) * 0.10)) if existing_ids else 0
            min_delete = max(1, int(len(existing_ids) * 0.05)) if existing_ids else 0
            actual_delete = min(len(ids_to_delete), max_delete)

            if actual_delete > 0:
                delete_ids = list(ids_to_delete)[:actual_delete]
                cursor.execute(f"DELETE FROM {table_name} WHERE id_applic IN ({','.join('?' * len(delete_ids))})",
                               delete_ids)
                stats['deleted'] += len(delete_ids)

            # 2. Добавление новых записей (≥10%)
            ids_to_add = new_ids - existing_ids
            if ids_to_add:
                add_data = [app for app in programs_data[op] if app['id_applic'] in ids_to_add]
                cursor.executemany(f'''
                    INSERT INTO {table_name} 
                    (id_applic, consent, priorit, physics_it_socre, russian_score, math_score, achivments_score, total_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', [
                    (a['id_applic'], a['consent'], a['priorit'], a['physics_it_socre'],
                     a['russian_score'], a['math_score'], a['achivments_score'], a['total_score'])
                    for a in add_data
                ])
                stats['added'] += len(add_data)

            # 3. Обновление существующих записей
            ids_to_update = existing_ids & new_ids
            if ids_to_update:
                update_data = [app for app in programs_data[op] if app['id_applic'] in ids_to_update]
                cursor.executemany(f'''
                    UPDATE {table_name}
                    SET consent = ?, priorit = ?, physics_it_socre = ?, russian_score = ?, 
                        math_score = ?, achivments_score = ?, total_score = ?
                    WHERE id_applic = ?
                ''', [
                    (a['consent'], a['priorit'], a['physics_it_socre'], a['russian_score'],
                     a['math_score'], a['achivments_score'], a['total_score'], a['id_applic'])
                    for a in update_data
                ])
                stats['updated'] += len(update_data)

            stats['total'] += len(programs_data[op])

        conn.commit()
        conn.close()

        duration = (datetime.now() - start_time).total_seconds()

        # Проверка требований к процентам
        validation_msg = self._validate_update_stats(stats, date)

        if duration > 5.0:
            return False, f"Время загрузки ({duration:.2f} сек) превышает лимит 5 сек", stats

        return True, f"Загрузка успешна за {duration:.2f} сек. {validation_msg}", stats

    def _validate_update_stats(self, stats: Dict, date: str) -> str:
        """Валидация соблюдения процентных требований при обновлении"""
        if stats['total'] == 0:
            return "Внимание: загружено 0 записей"

        deleted_pct = (stats['deleted'] / stats['total']) * 100 if stats['total'] > 0 else 0
        added_pct = (stats['added'] / stats['total']) * 100 if stats['total'] > 0 else 0

        msgs = []
        if stats['deleted'] > 0 and not (5 <= deleted_pct <= 10):
            msgs.append(f"Удалено {deleted_pct:.1f}% (требуется 5-10%)")
        if stats['added'] > 0 and added_pct < 10:
            msgs.append(f"Добавлено {added_pct:.1f}% (требуется ≥10%)")

        return "; ".join(msgs) if msgs else "Проценты обновления в норме"

    def validate_intersections(self, date: str) -> Tuple[bool, List[str]]:
        """
        Валидация пересечений множеств абитуриентов согласно ТЗ п. 2.9.
        Возвращает: (успех, список ошибок/предупреждений)
        """
        applicants = self.get_all_applicants(date)

        # Группировка абитуриентов по наборам программ
        intersections_2 = defaultdict(int)  # Пересечения 2 программ
        intersections_3 = defaultdict(int)  # Пересечения 3 программ
        intersections_4 = 0  # Пересечения 4 программ

        for app in applicants:
            programs = sorted([a['program'] for a in app['applications']])
            prog_count = len(programs)

            if prog_count == 2:
                key = f"{programs[0]}-{programs[1]}"
                intersections_2[key] += 1
            elif prog_count == 3:
                key = f"{programs[0]}-{programs[1]}-{programs[2]}"
                intersections_3[key] += 1
            elif prog_count == 4:
                intersections_4 += 1

        # Требуемые значения из ТЗ
        required_2 = {
            '01_08': {'ПМ-ИВТ': 22, 'ПМ-ИТСС': 17, 'ПМ-ИБ': 20, 'ИВТ-ИТСС': 19, 'ИВТ-ИБ': 22, 'ИТСС-ИБ': 17},
            '02_08': {'ПМ-ИВТ': 190, 'ПМ-ИТСС': 190, 'ПМ-ИБ': 150, 'ИВТ-ИТСС': 190, 'ИВТ-ИБ': 140, 'ИТСС-ИБ': 120},
            '03_08': {'ПМ-ИВТ': 760, 'ПМ-ИТСС': 600, 'ПМ-ИБ': 410, 'ИВТ-ИТСС': 750, 'ИВТ-ИБ': 460, 'ИТСС-ИБ': 500},
            '04_08': {'ПМ-ИВТ': 1090, 'ПМ-ИТСС': 1110, 'ПМ-ИБ': 1070, 'ИВТ-ИТСС': 1050, 'ИВТ-ИБ': 1040, 'ИТСС-ИБ': 1090}
        }

        required_3 = {
            '01_08': {'ПМ-ИВТ-ИТСС': 5, 'ПМ-ИВТ-ИБ': 5, 'ИВТ-ИТСС-ИБ': 5, 'ПМ-ИТСС-ИБ': 5, 'ПМ-ИВТ-ИТСС-ИБ': 3},
            '02_08': {'ПМ-ИВТ-ИТСС': 70, 'ПМ-ИВТ-ИБ': 70, 'ИВТ-ИТСС-ИБ': 70, 'ПМ-ИТСС-ИБ': 70, 'ПМ-ИВТ-ИТСС-ИБ': 50},
            '03_08': {'ПМ-ИВТ-ИТСС': 500, 'ПМ-ИВТ-ИБ': 260, 'ИВТ-ИТСС-ИБ': 300, 'ПМ-ИТСС-ИБ': 250,
                      'ПМ-ИВТ-ИТСС-ИБ': 200},
            '04_08': {'ПМ-ИВТ-ИТСС': 1020, 'ПМ-ИВТ-ИБ': 1020, 'ИВТ-ИТСС-ИБ': 1000, 'ПМ-ИТСС-ИБ': 1040,
                      'ПМ-ИВТ-ИТСС-ИБ': 1000}
        }

        errors = []

        # Проверка пересечений 2 программ
        for pair, required in required_2.get(date, {}).items():
            actual = intersections_2.get(pair, 0)
            if actual != required:
                errors.append(f"Пересечение {pair}: ожидается {required}, найдено {actual}")

        # Проверка пересечений 3 программ
        for triple, required in required_3.get(date, {}).items():
            if triple == 'ПМ-ИВТ-ИТСС-ИБ':
                actual = intersections_4
            else:
                actual = intersections_3.get(triple, 0)
            if actual != required:
                errors.append(f"Пересечение {triple}: ожидается {required}, найдено {actual}")

        return len(errors) == 0, errors

    def clear_database(self):
        """Полная очистка базы данных (для испытания №2.а)"""
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        for op in self.op_names:
            for date in self.dates:
                table_name = f"{op}_{date}"
                try:
                    cursor.execute(f"DELETE FROM {table_name}")
                except sqlite3.OperationalError:
                    # Таблица не существует - игнорируем
                    pass

        conn.commit()
        conn.close()
        return True


class TestDataGenerator:
    """Генератор тестовых данных, точно соответствующий требованиям ТЗ п. 2.8–2.9"""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        # Требуемое количество ЗАЯВЛЕНИЙ на каждую программу (п. 2.8 ТЗ)
        self.total_applications = {
            '01_08': {'ПМ': 60, 'ИВТ': 100, 'ИТСС': 50, 'ИБ': 70},
            '02_08': {'ПМ': 380, 'ИВТ': 370, 'ИТСС': 350, 'ИБ': 260},
            '03_08': {'ПМ': 1000, 'ИВТ': 1150, 'ИТСС': 1050, 'ИБ': 800},
            '04_08': {'ПМ': 1240, 'ИВТ': 1390, 'ИТСС': 1240, 'ИБ': 1190}
        }
        # Требуемое количество АБИТУРИЕНТОВ с МИНИМУМ указанными программами (п. 2.9 ТЗ)
        self.min_intersections_2 = {
            '01_08': {'ПМ-ИВТ': 22, 'ПМ-ИТСС': 17, 'ПМ-ИБ': 20, 'ИВТ-ИТСС': 19, 'ИВТ-ИБ': 22, 'ИТСС-ИБ': 17},
            '02_08': {'ПМ-ИВТ': 190, 'ПМ-ИТСС': 190, 'ПМ-ИБ': 150, 'ИВТ-ИТСС': 190, 'ИВТ-ИБ': 140, 'ИТСС-ИБ': 120},
            '03_08': {'ПМ-ИВТ': 760, 'ПМ-ИТСС': 600, 'ПМ-ИБ': 410, 'ИВТ-ИТСС': 750, 'ИВТ-ИБ': 460, 'ИТСС-ИБ': 500},
            '04_08': {'ПМ-ИВТ': 1090, 'ПМ-ИТСС': 1110, 'ПМ-ИБ': 1070, 'ИВТ-ИТСС': 1050, 'ИВТ-ИБ': 1040, 'ИТСС-ИБ': 1090}
        }
        self.min_intersections_3 = {
            '01_08': {'ПМ-ИВТ-ИТСС': 5, 'ПМ-ИВТ-ИБ': 5, 'ИВТ-ИТСС-ИБ': 5, 'ПМ-ИТСС-ИБ': 5},
            '02_08': {'ПМ-ИВТ-ИТСС': 70, 'ПМ-ИВТ-ИБ': 70, 'ИВТ-ИТСС-ИБ': 70, 'ПМ-ИТСС-ИБ': 70},
            '03_08': {'ПМ-ИВТ-ИТСС': 500, 'ПМ-ИВТ-ИБ': 260, 'ИВТ-ИТСС-ИБ': 300, 'ПМ-ИТСС-ИБ': 250},
            '04_08': {'ПМ-ИВТ-ИТСС': 1020, 'ПМ-ИВТ-ИБ': 1020, 'ИВТ-ИТСС-ИБ': 1000, 'ПМ-ИТСС-ИБ': 1040}
        }
        self.min_intersections_4 = {
            '01_08': 3,
            '02_08': 50,
            '03_08': 200,
            '04_08': 1000
        }

    def generate_csv(self, date: str, output_dir: str = ".") -> str:
        """Генерация CSV-файла с данными для указанной даты"""
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.join(output_dir, f"konkurs_{date}.csv")

        # Генерация абитуриентов
        applicants = self._generate_applicants(date)

        # Запись в CSV
        with open(filename, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['id_applic', 'consent', 'priorit', 'physics_it_socre', 'russian_score',
                             'math_score', 'achivments_score', 'total_score', 'program'])
            for app in applicants:
                for prog_data in app['programs_data']:
                    writer.writerow([
                        app['id'],
                        prog_data['consent'],
                        prog_data['priorit'],
                        prog_data['physics_it_socre'],
                        prog_data['russian_score'],
                        prog_data['math_score'],
                        prog_data['achivments_score'],
                        prog_data['total_score'],
                        prog_data['program']
                    ])
        total_apps = sum(len(a['programs_data']) for a in applicants)
        print(f"✓ Сгенерирован файл: {filename} ({len(applicants)} абитуриентов, {total_apps} заявлений)")
        return filename

    def _generate_applicants(self, date: str) -> List[Dict]:
        """Генерация абитуриентов с точным соблюдением требований ТЗ"""
        applicants = []
        current_id = 1

        # 1. Рассчитываем количество абитуриентов для КАЖДОГО ТИПА пересечения ("только")
        counts = self._calculate_exact_counts(date)

        # 2. Генерация абитуриентов с 4 программами
        for _ in range(counts['only_4']):
            applicants.append(self._create_applicant(current_id, date, ['ПМ', 'ИВТ', 'ИТСС', 'ИБ']))
            current_id += 1

        # 3. Генерация абитуриентов с 3 программами
        triples = [
            ('only_3_pm_ivt_its', ['ПМ', 'ИВТ', 'ИТСС']),
            ('only_3_pm_ivt_ib', ['ПМ', 'ИВТ', 'ИБ']),
            ('only_3_ivt_its_ib', ['ИВТ', 'ИТСС', 'ИБ']),
            ('only_3_pm_its_ib', ['ПМ', 'ИТСС', 'ИБ'])
        ]
        for key, progs in triples:
            for _ in range(counts[key]):
                applicants.append(self._create_applicant(current_id, date, progs))
                current_id += 1

        # 4. Генерация абитуриентов с 2 программами
        pairs = [
            ('only_2_pm_ivt', ['ПМ', 'ИВТ']),
            ('only_2_pm_its', ['ПМ', 'ИТСС']),
            ('only_2_pm_ib', ['ПМ', 'ИБ']),
            ('only_2_ivt_its', ['ИВТ', 'ИТСС']),
            ('only_2_ivt_ib', ['ИВТ', 'ИБ']),
            ('only_2_its_ib', ['ИТСС', 'ИБ'])
        ]
        for key, progs in pairs:
            for _ in range(counts[key]):
                applicants.append(self._create_applicant(current_id, date, progs))
                current_id += 1

        # 5. Генерация "чистых" абитуриентов (только одна программа)
        singles = [
            ('only_1_pm', ['ПМ']),
            ('only_1_ivt', ['ИВТ']),
            ('only_1_its', ['ИТСС']),
            ('only_1_ib', ['ИБ'])
        ]
        for key, progs in singles:
            for _ in range(counts[key]):
                applicants.append(self._create_applicant(current_id, date, progs))
                current_id += 1

        # 6. Для 04.08 гарантируем избыток согласий над местами (п. 2.11 ТЗ)
        if date == '04_08':
            self._ensure_agreements(applicants)

        # 7. Валидация
        self._validate_applicants(applicants, date)

        return applicants

    def _calculate_exact_counts(self, date: str) -> Dict[str, int]:
        """
        Рассчитывает ТОЧНОЕ количество абитуриентов для каждого типа пересечения ("только"),
        используя принцип включений-исключений.
        Возвращает словарь с ключами в формате 'only_X_...'.
        """
        min2 = self.min_intersections_2[date]
        min3 = self.min_intersections_3[date]
        min4 = self.min_intersections_4[date]
        total = self.total_applications[date]

        # 1. Только 4 программы
        only_4 = min4

        # 2. Только 3 программы (минимум_3 - только_4)
        only_3_pm_ivt_its = min3['ПМ-ИВТ-ИТСС'] - only_4
        only_3_pm_ivt_ib = min3['ПМ-ИВТ-ИБ'] - only_4
        only_3_ivt_its_ib = min3['ИВТ-ИТСС-ИБ'] - only_4
        only_3_pm_its_ib = min3['ПМ-ИТСС-ИБ'] - only_4

        # 3. Только 2 программы (минимум_2 - минимум_3(включающие пару) + только_4)
        only_2_pm_ivt = (
                min2['ПМ-ИВТ']
                - min3['ПМ-ИВТ-ИТСС']
                - min3['ПМ-ИВТ-ИБ']
                + only_4
        )
        only_2_pm_its = (
                min2['ПМ-ИТСС']
                - min3['ПМ-ИВТ-ИТСС']
                - min3['ПМ-ИТСС-ИБ']
                + only_4
        )
        only_2_pm_ib = (
                min2['ПМ-ИБ']
                - min3['ПМ-ИВТ-ИБ']
                - min3['ПМ-ИТСС-ИБ']
                + only_4
        )
        only_2_ivt_its = (
                min2['ИВТ-ИТСС']
                - min3['ПМ-ИВТ-ИТСС']
                - min3['ИВТ-ИТСС-ИБ']
                + only_4
        )
        only_2_ivt_ib = (
                min2['ИВТ-ИБ']
                - min3['ПМ-ИВТ-ИБ']
                - min3['ИВТ-ИТСС-ИБ']
                + only_4
        )
        only_2_its_ib = (
                min2['ИТСС-ИБ']
                - min3['ИВТ-ИТСС-ИБ']
                - min3['ПМ-ИТСС-ИБ']
                + only_4
        )

        # 4. Только 1 программа (формула включений-исключений)
        only_1_pm = (
                total['ПМ']
                - min2['ПМ-ИВТ'] - min2['ПМ-ИТСС'] - min2['ПМ-ИБ']
                + min3['ПМ-ИВТ-ИТСС'] + min3['ПМ-ИВТ-ИБ'] + min3['ПМ-ИТСС-ИБ']
                - only_4
        )
        only_1_ivt = (
                total['ИВТ']
                - min2['ПМ-ИВТ'] - min2['ИВТ-ИТСС'] - min2['ИВТ-ИБ']
                + min3['ПМ-ИВТ-ИТСС'] + min3['ПМ-ИВТ-ИБ'] + min3['ИВТ-ИТСС-ИБ']
                - only_4
        )
        only_1_its = (
                total['ИТСС']
                - min2['ПМ-ИТСС'] - min2['ИВТ-ИТСС'] - min2['ИТСС-ИБ']
                + min3['ПМ-ИВТ-ИТСС'] + min3['ИВТ-ИТСС-ИБ'] + min3['ПМ-ИТСС-ИБ']
                - only_4
        )
        only_1_ib = (
                total['ИБ']
                - min2['ПМ-ИБ'] - min2['ИВТ-ИБ'] - min2['ИТСС-ИБ']
                + min3['ПМ-ИВТ-ИБ'] + min3['ИВТ-ИТСС-ИБ'] + min3['ПМ-ИТСС-ИБ']
                - only_4
        )

        # Валидация неотрицательности
        counts_to_check = {
            'only_4': only_4,
            'only_3_pm_ivt_its': only_3_pm_ivt_its,
            'only_3_pm_ivt_ib': only_3_pm_ivt_ib,
            'only_3_ivt_its_ib': only_3_ivt_its_ib,
            'only_3_pm_its_ib': only_3_pm_its_ib,
            'only_2_pm_ivt': only_2_pm_ivt,
            'only_2_pm_its': only_2_pm_its,
            'only_2_pm_ib': only_2_pm_ib,
            'only_2_ivt_its': only_2_ivt_its,
            'only_2_ivt_ib': only_2_ivt_ib,
            'only_2_its_ib': only_2_its_ib,
            'only_1_pm': only_1_pm,
            'only_1_ivt': only_1_ivt,
            'only_1_its': only_1_its,
            'only_1_ib': only_1_ib
        }

        for key, value in counts_to_check.items():
            if value < 0:
                raise ValueError(
                    f"Отрицательное количество для '{key}': {value} на дату {date}. "
                    f"Проверьте корректность данных ТЗ."
                )

        return counts_to_check

    def _create_applicant(self, applicant_id: int, date: str, programs: List[str]) -> Dict:
        """Создание абитуриента с заявлениями на указанные программы"""
        # Профильные баллы в зависимости от программ
        math_base = 80
        physics_it_base = 80
        russian_base = 75

        # Корректировка под профиль программ
        if 'ПМ' in programs:
            math_base += 5
        if 'ИВТ' in programs or 'ИТСС' in programs:
            physics_it_base += 5
        if 'ИБ' in programs:
            physics_it_base -= 2

        math_score = random.randint(max(65, math_base - 15), 100)
        russian_score = random.randint(55, 95)
        physics_it_socre = random.randint(max(55, physics_it_base - 15), 100)
        achivments_score = random.choice([0, 0, 0, 2, 3, 5, 7, 10])
        total_score = math_score + russian_score + physics_it_socre + achivments_score

        # Приоритеты (рандомизируем порядок)
        priorities = list(range(1, len(programs) + 1))
        random.shuffle(priorities)

        # Вероятность согласия
        consent_chance = {
            '01_08': 0.3,
            '02_08': 0.6,
            '03_08': 0.8,
            '04_08': 0.95
        }

        # Для 04.08 высокобалльные абитуриенты почти всегда дают согласие
        consent_base = consent_chance[date]
        if date == '04_08' and total_score > 240:
            consent_base = 0.99

        programs_data = []
        for i, op in enumerate(programs):
            consent = 1 if random.random() < consent_base else 0
            programs_data.append({
                'program': op,
                'priorit': priorities[i],
                'consent': consent,
                'math_score': math_score,
                'russian_score': russian_score,
                'physics_it_socre': physics_it_socre,
                'achivments_score': achivments_score,
                'total_score': total_score
            })

        return {
            'id': applicant_id,
            'programs_data': programs_data
        }

    def _ensure_agreements(self, applicants: List[Dict]):
        """Гарантируем избыток согласий над местами для 04.08 (п. 2.11 ТЗ)"""
        places = {'ПМ': 40, 'ИВТ': 50, 'ИТСС': 30, 'ИБ': 20}
        agreements = {op: 0 for op in places.keys()}

        # Подсчёт текущих согласий
        for app in applicants:
            for prog in app['programs_data']:
                if prog['consent'] == 1:
                    agreements[prog['program']] += 1

        # Добавление недостающих согласий (гарантируем +15% от мест)
        for op in places.keys():
            required = int(places[op] * 1.15)
            if agreements[op] < required:
                # Находим абитуриентов с этой программой без согласия
                candidates = []
                for app in applicants:
                    for prog in app['programs_data']:
                        if prog['program'] == op and prog['consent'] == 0:
                            candidates.append((app, prog))

                # Сортируем по баллам
                candidates.sort(key=lambda x: x[1]['total_score'], reverse=True)

                # Добавляем согласия
                for i in range(min(required - agreements[op], len(candidates))):
                    candidates[i][1]['consent'] = 1
                    agreements[op] += 1

    def _validate_applicants(self, applicants: List[Dict], date: str):
        """Валидация соответствия сгенерированных данных требованиям ТЗ"""
        # Подсчёт заявлений по программам
        app_counts = {'ПМ': 0, 'ИВТ': 0, 'ИТСС': 0, 'ИБ': 0}
        for app in applicants:
            for prog in app['programs_data']:
                app_counts[prog['program']] += 1

        # Проверка общего количества заявлений
        expected = self.total_applications[date]
        for op in app_counts.keys():
            if app_counts[op] != expected[op]:
                raise ValueError(
                    f"Несоответствие количества заявлений для {op} на {date}: "
                    f"ожидалось {expected[op]}, получено {app_counts[op]}"
                )

        # Подсчёт пересечений абитуриентов (с фиксированным порядком программ как в ТЗ)
        intersections_2 = defaultdict(int)
        intersections_3 = defaultdict(int)
        intersections_4 = 0

        for app in applicants:
            progs = [p['program'] for p in app['programs_data']]
            prog_count = len(progs)

            if prog_count == 4:
                intersections_4 += 1
            elif prog_count == 3:
                # Формируем ключ в порядке ТЗ: ПМ > ИВТ > ИТСС > ИБ
                ordered = []
                if 'ПМ' in progs: ordered.append('ПМ')
                if 'ИВТ' in progs: ordered.append('ИВТ')
                if 'ИТСС' in progs: ordered.append('ИТСС')
                if 'ИБ' in progs: ordered.append('ИБ')
                key = f"{ordered[0]}-{ordered[1]}-{ordered[2]}"
                intersections_3[key] += 1
            elif prog_count == 2:
                # Формируем ключ в порядке ТЗ
                ordered = []
                if 'ПМ' in progs: ordered.append('ПМ')
                if 'ИВТ' in progs: ordered.append('ИВТ')
                if 'ИТСС' in progs: ordered.append('ИТСС')
                if 'ИБ' in progs: ordered.append('ИБ')
                key = f"{ordered[0]}-{ordered[1]}"
                intersections_2[key] += 1

        # Проверка пересечения 4 программ
        if intersections_4 != self.min_intersections_4[date]:
            raise ValueError(
                f"Пересечение 4 программ на {date}: "
                f"ожидалось {self.min_intersections_4[date]}, получено {intersections_4}"
            )

        # Проверка пересечений 3 программ (минимум = только_3 + только_4)
        for triple_key, expected in self.min_intersections_3[date].items():
            actual = intersections_3.get(triple_key, 0) + intersections_4
            if actual != expected:
                raise ValueError(
                    f"Пересечение {triple_key} на {date}: "
                    f"ожидалось {expected}, получено {actual} "
                    f"(только_3={intersections_3.get(triple_key, 0)}, только_4={intersections_4})"
                )

        # Проверка пересечений 2 программ (минимум = только_2 + только_3(включающие пару) + только_4)
        for pair_key, expected in self.min_intersections_2[date].items():
            actual = intersections_2.get(pair_key, 0)

            # Добавляем пересечения 3 программ, включающие эту пару
            if pair_key == 'ПМ-ИВТ':
                actual += intersections_3.get('ПМ-ИВТ-ИТСС', 0) + intersections_3.get('ПМ-ИВТ-ИБ', 0)
            elif pair_key == 'ПМ-ИТСС':
                actual += intersections_3.get('ПМ-ИВТ-ИТСС', 0) + intersections_3.get('ПМ-ИТСС-ИБ', 0)
            elif pair_key == 'ПМ-ИБ':
                actual += intersections_3.get('ПМ-ИВТ-ИБ', 0) + intersections_3.get('ПМ-ИТСС-ИБ', 0)
            elif pair_key == 'ИВТ-ИТСС':
                actual += intersections_3.get('ПМ-ИВТ-ИТСС', 0) + intersections_3.get('ИВТ-ИТСС-ИБ', 0)
            elif pair_key == 'ИВТ-ИБ':
                actual += intersections_3.get('ПМ-ИВТ-ИБ', 0) + intersections_3.get('ИВТ-ИТСС-ИБ', 0)
            elif pair_key == 'ИТСС-ИБ':
                actual += intersections_3.get('ИВТ-ИТСС-ИБ', 0) + intersections_3.get('ПМ-ИТСС-ИБ', 0)

            # Добавляем пересечение 4 программ
            actual += intersections_4

            if actual != expected:
                raise ValueError(
                    f"Пересечение {pair_key} на {date}: "
                    f"ожидалось {expected}, получено {actual}"
                )

        print(f"✓ Валидация данных для {date} пройдена успешно")
class AdmissionCalculator:
    """Расчет проходных баллов с корректным учетом приоритетов"""

    def __init__(self, db_manager):
        self.db = db_manager
        self.places = db_manager.places

    def calculate_passing_scores(self, date):
        """
        Алгоритм расчета проходных баллов (согласно ТЗ):
        1. Собрать всех абитуриентов со всеми заявлениями за дату
        2. Отфильтровать только абитуриентов с хотя бы одним согласием
        3. Отсортировать абитуриентов по сумме баллов (убывание)
        4. Итеративно распределять абитуриентов по приоритетам:
           - Для каждого абитуриента в порядке убывания баллов
           - Проверяем его программы в порядке приоритета (1 -> 4)
           - Если на программе есть свободные места И абитуриент подал согласие на эту программу - зачисляем
           - Абитуриент зачисляется ТОЛЬКО на ОДНУ программу (первую подходящую по приоритету)
        5. Проходной балл = минимальный балл среди зачисленных на программу
           (если зачисленных меньше, чем мест -> "НЕДОБОР")
        """
        applicants = self.db.get_all_applicants(date)

        # Фильтрация: только абитуриенты с хотя бы одним согласием
        applicants_with_agreement = [
            app for app in applicants
            if any(a['agreement'] for a in app['applications'])
        ]

        # Сортировка по баллам (убывание)
        applicants_with_agreement.sort(key=lambda x: x['total_score'], reverse=True)

        # Инициализация списков зачисленных для каждой программы
        enrolled = {op: [] for op in self.places.keys()}
        occupied_places = {op: 0 for op in self.places.keys()}

        # Распределение абитуриентов по приоритетам
        for app in applicants_with_agreement:
            # Сортируем заявления абитуриента по приоритету (только с согласием)
            applications_sorted = sorted(
                [a for a in app['applications'] if a['agreement']],
                key=lambda x: x['priority']
            )

            # Пробуем зачислить на первую подходящую программу по приоритету
            for app_data in applications_sorted:
                program = app_data['program']
                if occupied_places[program] < self.places[program]:
                    enrolled[program].append({
                        'id': app['id'],
                        'total_score': app['total_score'],
                        'priority': app_data['priority'],
                        'math_score': app['math_score'],
                        'russian_score': app['russian_score'],
                        'physics_it_socre': app['physics_it_socre'],
                        'achivments_score': app['achivments_score']
                    })
                    occupied_places[program] += 1
                    break  # Зачислен только на одну программу

        # Расчет проходных баллов
        passing_scores = {}
        for op in self.places.keys():
            if occupied_places[op] == 0:
                passing_scores[op] = 'НЕДОБОР'
            elif occupied_places[op] < self.places[op]:
                passing_scores[op] = 'НЕДОБОР'
            else:
                # Сортируем зачисленных по баллам для определения проходного
                enrolled[op].sort(key=lambda x: x['total_score'], reverse=True)
                passing_scores[op] = enrolled[op][self.places[op] - 1]['total_score']

        return passing_scores, enrolled, occupied_places

    def get_statistics(self, date, enrolled_lists):
        """Расчет статистики по программам"""
        stats = {op: {
            'total_applications': 0,
            'places': self.places[op],
            'priority_1': 0, 'priority_2': 0, 'priority_3': 0, 'priority_4': 0,
            'enrolled_p1': 0, 'enrolled_p2': 0, 'enrolled_p3': 0, 'enrolled_p4': 0,
            'agreements': 0
        } for op in self.places.keys()}

        # Статистика по всем заявлениям
        for op in self.places.keys():
            program_list = self.db.get_program_list(op, date)
            stats[op]['total_applications'] = len(program_list)
            stats[op]['agreements'] = sum(1 for app in program_list if app['consent'])

            for applicant in program_list:
                prio = applicant['priorit']
                if 1 <= prio <= 4:
                    stats[op][f'priority_{prio}'] += 1

        # Статистика по зачисленным
        for op, enrolled in enrolled_lists.items():
            for app in enrolled:
                prio = app['priority']
                if 1 <= prio <= 4:
                    stats[op][f'enrolled_p{prio}'] += 1

        return stats


class ReportGenerator:
    """Генерация отчетов в PDF"""

    def __init__(self, db_manager, calculator):
        self.db = db_manager
        self.calc = calculator
        self.passing_scores_history = {op: [] for op in ['ПМ', 'ИВТ', 'ИТСС', 'ИБ']}

    def save_passing_scores(self, date, passing_scores):
        """Сохранение проходных баллов для построения динамики"""
        for op in ['ПМ', 'ИВТ', 'ИТСС', 'ИБ']:
            score = passing_scores[op] if isinstance(passing_scores[op], int) else 0
            self.passing_scores_history[op].append((date, score))

    def generate_report(self, date, filename=None):
        """Генерация полного отчета за указанную дату"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"otchet_{date}_{timestamp}.pdf"

        passing_scores, enrolled_lists, occupied_places = self.calc.calculate_passing_scores(date)
        stats = self.calc.get_statistics(date, enrolled_lists)

        # Сохраняем для истории
        self.save_passing_scores(date, passing_scores)

        # Создание PDF
        doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
        styles = getSampleStyleSheet()
        for style in styles.byName.values():
            style.fontName = 'DejaVuSans'  # Применяем шрифт ко всем стилям

        styles.add(ParagraphStyle(
            name='Center',
            alignment=1,
            fontSize=12,
            fontName='DejaVuSans'
        ))
        story = []

        # Заголовок
        title = Paragraph("ОТЧЕТ ПО РЕЗУЛЬТАТАМ ПРИЕМНОЙ КАМПАНИИ", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        story.append(Paragraph(f"Дата конкурсного списка: {date.replace('_', '.')}", styles['Heading2']))
        story.append(Spacer(1, 12))

        # Дата формирования отчета
        report_time = Paragraph(
            f"Дата и время формирования отчета: {datetime.now().strftime('%d.%m.%Y %H:%M:%S')}",
            styles['Normal']
        )
        story.append(report_time)
        story.append(Spacer(1, 24))

        # Проходные баллы
        story.append(Paragraph("Проходные баллы по образовательным программам:", styles['Heading2']))
        story.append(Spacer(1, 12))

        score_data = [['Образовательная программа', 'Количество мест', 'Заполнено мест', 'Проходной балл']]
        for op in ['ПМ', 'ИВТ', 'ИТСС', 'ИБ']:
            score = passing_scores[op]
            score_str = str(score) if isinstance(score, int) else score
            score_data.append([
                op,
                str(self.calc.places[op]),
                f"{occupied_places[op]}/{self.calc.places[op]}",
                score_str
            ])

        score_table = Table(score_data, colWidths=[150, 100, 100, 120])
        score_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),  # Заголовки — жирный
            ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, 0), 8),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
        ]))
        story.append(score_table)
        story.append(Spacer(1, 24))

        # Списки зачисленных
        for op in ['ПМ', 'ИВТ', 'ИТСС', 'ИБ']:
            story.append(Paragraph(f"Зачисленные на {op} ({occupied_places[op]}/{self.calc.places[op]} мест):",
                                   styles['Heading3']))
            story.append(Spacer(1, 6))

            if not enrolled_lists[op]:
                story.append(Paragraph("НЕДОБОР", styles['Normal']))
            else:
                # Сортируем по баллам для отображения
                enrolled_sorted = sorted(enrolled_lists[op], key=lambda x: x['total_score'], reverse=True)
                enrolled_data = [['ID абитуриента', 'Сумма баллов', 'Приоритет', 'Матем.', 'Русский', 'Физ/ИКТ', 'ИД']]
                for app in enrolled_sorted[:min(30, len(enrolled_sorted))]:
                    enrolled_data.append([
                        str(app['id']),
                        str(app['total_score']),
                        str(app['priority']),
                        str(app['math_score']),
                        str(app['russian_score']),
                        str(app['physics_it_socre']),
                        str(app['achivments_score'])
                    ])

                if len(enrolled_sorted) > 30:
                    enrolled_data.append(['...', '...', '...', '...', '...', '...', '...'])

                enrolled_table = Table(enrolled_data, colWidths=[80, 80, 70, 50, 50, 50, 40])
                enrolled_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),  # Заголовки — жирный
                    ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTSIZE', (0, 0), (-1, 0), 8),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('FONTSIZE', (0, 1), (-1, -1), 8),
                ]))
                story.append(enrolled_table)

            story.append(Spacer(1, 18))

        # Статистика
        story.append(Paragraph("Статистика по образовательным программам:", styles['Heading2']))
        story.append(Spacer(1, 12))

        stat_data = [
            ['', 'ПМ', 'ИВТ', 'ИТСС', 'ИБ'],
            ['Общее кол-во заявлений',
             str(stats['ПМ']['total_applications']),
             str(stats['ИВТ']['total_applications']),
             str(stats['ИТСС']['total_applications']),
             str(stats['ИБ']['total_applications'])],
            ['Количество мест на ОП', '40', '50', '30', '20'],
            ['Кол-во согласий',
             str(stats['ПМ']['agreements']),
             str(stats['ИВТ']['agreements']),
             str(stats['ИТСС']['agreements']),
             str(stats['ИБ']['agreements'])],
            ['Кол-во заявлений 1-го приоритета',
             str(stats['ПМ']['priority_1']),
             str(stats['ИВТ']['priority_1']),
             str(stats['ИТСС']['priority_1']),
             str(stats['ИБ']['priority_1'])],
            ['Кол-во заявлений 2-го приоритета',
             str(stats['ПМ']['priority_2']),
             str(stats['ИВТ']['priority_2']),
             str(stats['ИТСС']['priority_2']),
             str(stats['ИБ']['priority_2'])],
            ['Кол-во заявлений 3-го приоритета',
             str(stats['ПМ']['priority_3']),
             str(stats['ИВТ']['priority_3']),
             str(stats['ИТСС']['priority_3']),
             str(stats['ИБ']['priority_3'])],
            ['Кол-во заявлений 4-го приоритета',
             str(stats['ПМ']['priority_4']),
             str(stats['ИВТ']['priority_4']),
             str(stats['ИТСС']['priority_4']),
             str(stats['ИБ']['priority_4'])],
            ['Кол-во зачисленных 1-го приоритета',
             str(stats['ПМ']['enrolled_p1']),
             str(stats['ИВТ']['enrolled_p1']),
             str(stats['ИТСС']['enrolled_p1']),
             str(stats['ИБ']['enrolled_p1'])],
            ['Кол-во зачисленных 2-го приоритета',
             str(stats['ПМ']['enrolled_p2']),
             str(stats['ИВТ']['enrolled_p2']),
             str(stats['ИТСС']['enrolled_p2']),
             str(stats['ИБ']['enrolled_p2'])],
            ['Кол-во зачисленных 3-го приоритета',
             str(stats['ПМ']['enrolled_p3']),
             str(stats['ИВТ']['enrolled_p3']),
             str(stats['ИТСС']['enrolled_p3']),
             str(stats['ИБ']['enrolled_p3'])],
            ['Кол-во зачисленных 4-го приоритета',
             str(stats['ПМ']['enrolled_p4']),
             str(stats['ИВТ']['enrolled_p4']),
             str(stats['ИТСС']['enrolled_p4']),
             str(stats['ИБ']['enrolled_p4'])],
        ]

        stat_table = Table(stat_data, colWidths=[200, 80, 80, 80, 80])
        stat_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'DejaVuSans-Bold'),  # Заголовки — жирный
            ('FONTNAME', (0, 1), (-1, -1), 'DejaVuSans'),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        story.append(stat_table)
        story.append(Spacer(1, 24))

        # График динамики проходных баллов
        if any(len(scores) > 1 for scores in self.passing_scores_history.values()):
            story.append(Paragraph("Динамика проходных баллов по дням приемной кампании:", styles['Heading2']))
            story.append(Spacer(1, 12))
            self._add_passing_score_chart(story)

        # Генерация PDF
        try:
            doc.build(story)
            print(f"✓ Отчет успешно сохранен: {filename}")
            return filename
        except Exception as e:
            print(f"✗ Ошибка при создании отчета: {e}")
            return None

    def _add_passing_score_chart(self, story):
        """Добавление графика динамики проходных баллов в отчет"""
        dates_labels = ['01.08', '02.08', '03.08', '04.08']
        colors_map = {'ПМ': 'blue', 'ИВТ': 'green', 'ИТСС': 'red', 'ИБ': 'purple'}

        fig, ax = plt.subplots(figsize=(8, 5))

        for op in ['ПМ', 'ИВТ', 'ИТСС', 'ИБ']:
            scores = [score for _, score in self.passing_scores_history[op]]
            if len(scores) > 0:
                ax.plot(dates_labels[:len(scores)], scores,
                        marker='o', label=op, color=colors_map[op], linewidth=2, markersize=8)

        ax.set_xlabel('Дата', fontsize=12, fontweight='bold')
        ax.set_ylabel('Проходной балл', fontsize=12, fontweight='bold')
        ax.set_title('Динамика проходных баллов по образовательным программам', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)

        # Сохранение графика
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        plt.close(fig)
        img_buffer.seek(0)

        img_filename = f'temp_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        with open(img_filename, 'wb') as f:
            f.write(img_buffer.read())

        story.append(Image(img_filename, width=480, height=300))
        story.append(Spacer(1, 24))

        try:
            os.remove(img_filename)
        except:
            pass


class AdmissionGUI:
    """Графический интерфейс для анализа поступления"""

    def __init__(self, root):
        self.root = root
        self.root.title("Анализ поступления | Московская предпрофессиональная олимпиада")
        self.root.geometry("1200x850")
        self.root.minsize(1000, 750)

        self.db = DatabaseManager('predprof_vit.db')
        self.calculator = AdmissionCalculator(self.db)
        self.report_gen = ReportGenerator(self.db, self.calculator)
        self.test_gen = TestDataGenerator(self.db)

        self.current_date = '04_08'
        self.current_program = 'ПМ'
        self.passing_scores_cache = {}

        self.create_widgets()
        self.load_data()

    def create_widgets(self):
        # Верхняя панель управления
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)

        # Группа: Загрузка данных
        load_group = ttk.LabelFrame(control_frame, text="Загрузка данных", padding=5)
        load_group.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Button(load_group, text="Очистить БД",
                   command=self.clear_database, width=12).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(load_group, text="Загрузить CSV",
                   command=self.load_csv_file, width=15).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(load_group, text="Сгенерировать тестовые данные",
                   command=self.generate_test_data, width=25).pack(side=tk.LEFT)

        # Группа: Анализ
        analysis_group = ttk.LabelFrame(control_frame, text="Анализ", padding=5)
        analysis_group.pack(side=tk.LEFT, padx=(0, 15))

        ttk.Label(analysis_group, text="Дата:").pack(side=tk.LEFT, padx=(0, 5))
        self.date_combo = ttk.Combobox(analysis_group, values=['01_08', '02_08', '03_08', '04_08'],
                                       width=10, state='readonly')
        self.date_combo.set('04_08')
        self.date_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.date_combo.bind('<<ComboboxSelected>>', self.on_date_change)

        ttk.Label(analysis_group, text="Программа:").pack(side=tk.LEFT, padx=(0, 5))
        self.program_combo = ttk.Combobox(analysis_group, values=['ПМ', 'ИВТ', 'ИТСС', 'ИБ'],
                                          width=10, state='readonly')
        self.program_combo.set('ПМ')
        self.program_combo.pack(side=tk.LEFT, padx=(0, 15))
        self.program_combo.bind('<<ComboboxSelected>>', self.on_program_change)

        ttk.Button(analysis_group, text="Рассчитать проходные баллы",
                   command=self.calculate_passing_scores, width=25).pack(side=tk.LEFT)

        # Группа: Отчеты
        report_group = ttk.LabelFrame(control_frame, text="Отчеты", padding=5)
        report_group.pack(side=tk.LEFT, padx=(0, 15))
        ttk.Button(report_group, text="Сформировать отчет (PDF)",
                   command=self.generate_report, width=25).pack()

        # Панель проходных баллов
        self.scores_frame = ttk.LabelFrame(self.root, text="Проходные баллы", padding=10)
        self.scores_frame.pack(fill=tk.X, padx=10, pady=5)

        self.scores_labels = {}
        scores_inner = ttk.Frame(self.scores_frame)
        scores_inner.pack(fill=tk.X)

        for op in ['ПМ', 'ИВТ', 'ИТСС', 'ИБ']:
            frame = ttk.Frame(scores_inner)
            frame.pack(side=tk.LEFT, padx=15)
            ttk.Label(frame, text=f"{op}:", font=('Arial', 12, 'bold')).pack()
            label = ttk.Label(frame, text="--", font=('Arial', 14, 'bold'), foreground='blue')
            label.pack()
            self.scores_labels[op] = label

        # Таблица абитуриентов
        table_frame = ttk.LabelFrame(self.root, text="Конкурсный список", padding=10)
        table_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Панель фильтров
        filter_frame = ttk.Frame(table_frame)
        filter_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(filter_frame, text="Фильтр по согласию:").pack(side=tk.LEFT, padx=(0, 5))
        self.agreement_filter = ttk.Combobox(filter_frame,
                                             values=['Все', 'Только с согласием', 'Без согласия'], width=20,
                                             state='readonly')
        self.agreement_filter.set('Все')
        self.agreement_filter.pack(side=tk.LEFT, padx=(0, 20))
        self.agreement_filter.bind('<<ComboboxSelected>>', self.apply_filters)

        ttk.Label(filter_frame, text="Сортировка по:").pack(side=tk.LEFT, padx=(0, 5))
        self.sort_combo = ttk.Combobox(filter_frame,
                                       values=['Балл (убыв)', 'Балл (возр)', 'Приоритет', 'ID'], width=15,
                                       state='readonly')
        self.sort_combo.set('Балл (убыв)')
        self.sort_combo.pack(side=tk.LEFT, padx=(0, 20))
        self.sort_combo.bind('<<ComboboxSelected>>', self.apply_filters)

        ttk.Button(filter_frame, text="Применить фильтры", command=self.apply_filters).pack(side=tk.LEFT)

        # Создание таблицы
        self.tree = ttk.Treeview(table_frame, columns=(
            'id_applic', 'consent', 'priorit', 'math_score', 'russian_score', 'physics_it_socre', 'achivments_score',
            'total_score'
        ), show='headings', height=15)

        # Настройка колонок (с правильными названиями)
        self.tree.heading('id_applic', text='ID')
        self.tree.heading('consent', text='Согласие')
        self.tree.heading('priorit', text='Приоритет')
        self.tree.heading('math_score', text='Матем.')
        self.tree.heading('russian_score', text='Русский')
        self.tree.heading('physics_it_socre', text='Физ/ИКТ')
        self.tree.heading('achivments_score', text='ИД')
        self.tree.heading('total_score', text='Сумма')

        self.tree.column('id_applic', width=80, anchor='center')
        self.tree.column('consent', width=80, anchor='center')
        self.tree.column('priorit', width=100, anchor='center')
        self.tree.column('math_score', width=80, anchor='center')
        self.tree.column('russian_score', width=80, anchor='center')
        self.tree.column('physics_it_socre', width=80, anchor='center')
        self.tree.column('achivments_score', width=60, anchor='center')
        self.tree.column('total_score', width=80, anchor='center')

        # Скроллбары
        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        hsb.pack(side=tk.BOTTOM, fill=tk.X)

        # Статусная строка
        self.status_var = tk.StringVar()
        self.status_var.set("Готово к работе")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def clear_database(self):
        """Очистка базы данных"""
        if messagebox.askyesno("Подтверждение", "Вы уверены, что хотите очистить всю базу данных?"):
            self.db.clear_database()
            self.status_var.set("База данных очищена")
            # Очистка таблицы
            for item in self.tree.get_children():
                self.tree.delete(item)
            messagebox.showinfo("Успех", "База данных успешно очищена")

    def load_csv_file(self):
        """Загрузка данных из CSV-файла"""
        filename = filedialog.askopenfilename(
            title="Выберите CSV-файл с конкурсными списками",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if not filename:
            return

        # Определение даты из имени файла или выбор пользователем
        date = self.date_combo.get()

        self.status_var.set(f"Загрузка данных из {os.path.basename(filename)}...")
        self.root.update()

        success, message, stats = self.db.import_from_csv(filename, date)

        if success:
            self.status_var.set(f"✓ {message}")
            self.load_data()
            messagebox.showinfo("Успех",
                                f"Данные успешно загружены!\n{message}\n\nСтатистика:\nУдалено: {stats['deleted']}\nДобавлено: {stats['added']}\nОбновлено: {stats['updated']}")
        else:
            self.status_var.set(f"✗ {message}")
            messagebox.showerror("Ошибка", f"Ошибка загрузки:\n{message}")

    def generate_test_data(self):
        """Генерация тестовых данных для всех 4 дней"""
        if messagebox.askyesno("Подтверждение",
                               "Эта операция сгенерирует тестовые данные для всех 4 дней (01.08-04.08) и загрузит их в БД. Продолжить?"):

            self.status_var.set("Генерация тестовых данных...")
            self.root.update()

            try:
                # Генерация и загрузка для каждого дня
                for date in ['01_08', '02_08', '03_08', '04_08']:
                    self.status_var.set(f"Генерация данных для {date.replace('_', '.')}...")
                    self.root.update()

                    csv_file = self.test_gen.generate_csv(date, "test_data")

                    # Загрузка в БД
                    success, message, _ = self.db.import_from_csv(csv_file, date)
                    if not success:
                        raise Exception(f"Ошибка загрузки {date}: {message}")

                # Валидация пересечений для последнего дня
                self.status_var.set("Валидация пересечений...")
                self.root.update()

                is_valid, errors = self.db.validate_intersections('04_08')
                if not is_valid:
                    messagebox.showwarning("Предупреждение",
                                           f"Обнаружены несоответствия в пересечениях:\n" + "\n".join(errors[:5]))

                self.status_var.set("Тестовые данные успешно сгенерированы и загружены")
                self.load_data()
                messagebox.showinfo("Успех",
                                    "Тестовые данные для всех 4 дней успешно сгенерированы и загружены в БД!\n"
                                    "Данные соответствуют требованиям ТЗ по количеству абитуриентов и пересечениям.")
            except Exception as e:
                self.status_var.set(f"Ошибка генерации: {str(e)}")
                messagebox.showerror("Ошибка", f"Ошибка при генерации тестовых данных:\n{str(e)}")

    def load_data(self):
        """Загрузка данных в таблицу"""
        self.status_var.set("Загрузка данных...")
        self.root.update()

        start_time = datetime.now()
        applicants = self.db.get_program_list(self.current_program, self.current_date)
        load_time = (datetime.now() - start_time).total_seconds()

        # Очистка таблицы
        for item in self.tree.get_children():
            self.tree.delete(item)

        # Заполнение таблицы
        for app in applicants:
            self.tree.insert('', 'end', values=(
                app['id_applic'],
                '✓' if app['consent'] else '✗',
                app['priorit'],
                app['math_score'],
                app['russian_score'],
                app['physics_it_socre'],
                app['achivments_score'],
                app['total_score']
            ))

        self.status_var.set(
            f"Загружено {len(applicants)} абитуриентов для {self.current_program} на {self.current_date.replace('_', '.')} (время: {load_time:.2f} сек)")

    def apply_filters(self, event=None):
        """Применение фильтров и сортировки к таблице"""
        self.status_var.set("Применение фильтров...")
        self.root.update()

        start_time = datetime.now()
        applicants = self.db.get_program_list(self.current_program, self.current_date)

        # Фильтрация по согласию
        filter_type = self.agreement_filter.get()
        if filter_type == 'Только с согласием':
            applicants = [a for a in applicants if a['consent']]
        elif filter_type == 'Без согласия':
            applicants = [a for a in applicants if not a['consent']]

        # Сортировка
        sort_type = self.sort_combo.get()
        if sort_type == 'Балл (убыв)':
            applicants.sort(key=lambda x: x['total_score'], reverse=True)
        elif sort_type == 'Балл (возр)':
            applicants.sort(key=lambda x: x['total_score'])
        elif sort_type == 'Приоритет':
            applicants.sort(key=lambda x: x['priorit'])
        elif sort_type == 'ID':
            applicants.sort(key=lambda x: x['id_applic'])

        # Обновление таблицы
        for item in self.tree.get_children():
            self.tree.delete(item)

        for app in applicants:
            self.tree.insert('', 'end', values=(
                app['id_applic'],
                '✓' if app['consent'] else '✗',
                app['priorit'],
                app['math_score'],
                app['russian_score'],
                app['physics_it_socre'],
                app['achivments_score'],
                app['total_score']
            ))

        filter_time = (datetime.now() - start_time).total_seconds()
        self.status_var.set(f"Применены фильтры: {len(applicants)} записей (время: {filter_time:.2f} сек)")

        if filter_time > 3.0:
            messagebox.showwarning("Предупреждение",
                                   f"Время обновления визуализации ({filter_time:.2f} сек) превышает требование ТЗ (< 3 сек)")

    def calculate_passing_scores(self):
        """Расчет и отображение проходных баллов"""
        self.status_var.set("Расчет проходных баллов...")
        self.root.update()

        start_time = datetime.now()
        passing_scores, _, _ = self.calculator.calculate_passing_scores(self.current_date)
        calc_time = (datetime.now() - start_time).total_seconds()

        # Обновление меток проходных баллов
        for op, score in passing_scores.items():
            score_str = str(score) if isinstance(score, int) else score
            color = 'green' if isinstance(score, int) else 'red'
            self.scores_labels[op].config(text=score_str, foreground=color)

        self.passing_scores_cache[self.current_date] = passing_scores

        self.status_var.set(
            f"Проходные баллы рассчитаны за {calc_time:.2f} сек | {self.current_date.replace('_', '.')}")

        # Демонстрация распределения по приоритетам для испытания №2
        detailed_msg = "Расчет проходных баллов выполнен:\n"
        for op in ['ПМ', 'ИВТ', 'ИТСС', 'ИБ']:
            detailed_msg += f"{op}: {passing_scores[op]}\n"

        messagebox.showinfo("Расчет завершен", detailed_msg)

    def generate_report(self):
        """Генерация отчета в PDF"""
        if self.current_date not in self.passing_scores_cache:
            self.calculate_passing_scores()

        self.status_var.set("Генерация отчета в PDF...")
        self.root.update()

        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF files", "*.pdf")],
                initialfile=f"otchet_{self.current_date}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            )

            if filename:
                threading.Thread(target=self._generate_report_thread, args=(filename,), daemon=True).start()
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка при генерации отчета: {e}")
            self.status_var.set("Ошибка генерации отчета")

    def _generate_report_thread(self, filename):
        try:
            report_file = self.report_gen.generate_report(self.current_date, filename)
            if report_file:
                self.root.after(0, lambda: messagebox.showinfo("Успех", f"Отчет успешно сохранен:\n{report_file}"))
                self.root.after(0, lambda: self.status_var.set(f"Отчет сохранен: {report_file}"))
            else:
                self.root.after(0, lambda: messagebox.showerror("Ошибка", "Не удалось создать отчет"))
                self.root.after(0, lambda: self.status_var.set("Ошибка создания отчета"))
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Ошибка", f"Ошибка при генерации отчета: {e}"))
            self.root.after(0, lambda: self.status_var.set(f"Ошибка: {e}"))

    def on_date_change(self, event=None):
        self.current_date = self.date_combo.get()
        self.load_data()

    def on_program_change(self, event=None):
        self.current_program = self.program_combo.get()
        self.load_data()


def main():
    """Точка входа в программу"""
    print("=" * 70)
    print("СИСТЕМА АНАЛИЗА ПОСТУПЛЕНИЯ")
    print("Московская предпрофессиональная олимпиада школьников")
    print("=" * 70)
    print("\nПроверка зависимостей...")

    # Проверка зависимостей
    missing = []
    try:
        import pandas
    except ImportError:
        missing.append("pandas")

    try:
        import reportlab
    except ImportError:
        missing.append("reportlab")

    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")

    if missing:
        print(f"✗ Отсутствуют необходимые библиотеки: {', '.join(missing)}")
        print("\nУстановите зависимости командой:")
        print("pip install pandas reportlab matplotlib")
        return

    print("✓ Все зависимости установлены")
    print("\nСтруктура базы данных 'predprof_vit.db':")
    print("  Таблицы: ПМ_01_08, ПМ_02_08, ПМ_03_08, ПМ_04_08, ...")
    print("  Столбцы в каждой таблице:")
    print("    • id_applic INTEGER PRIMARY KEY")
    print("    • consent BOOLEAN NOT NULL")
    print("    • priorit INTEGER NOT NULL")
    print("    • physics_it_socre INTEGER NOT NULL  (обратите внимание на опечатку 'socre')")
    print("    • russian_score INTEGER NOT NULL")
    print("    • math_score INTEGER NOT NULL")
    print("    • achivments_score INTEGER NOT NULL  (обратите внимание на опечатку 'achivments')")
    print("    • total_score INTEGER NOT NULL")
    print("\nЗапуск графического интерфейса...")
    print("=" * 70)

    root = tk.Tk()
    app = AdmissionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()