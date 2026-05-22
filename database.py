import psycopg2
from db_config import DB_CONFIG
from datetime import datetime


class ContactCenterDB:
    def __init__(self):
        self.conn = None
        self.cursor = None

    def connect(self):
        """Установить соединение с БД"""
        try:
            self.conn = psycopg2.connect(**DB_CONFIG)
            self.cursor = self.conn.cursor()
            print("✅ Подключено к contact_center")
        except Exception as e:
            print(f"❌ Ошибка подключения: {e}")
            raise

    def disconnect(self):
        """Закрыть соединение"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        print("🔌 Соединение закрыто")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    # Методы операторов

    def get_all_operators(self):
        """Получить всех операторов"""
        self.cursor.execute("""
            SELECT 
                id, full_name, gender, birth_date, start_date,
                patience_level, stress_resistance_level, empathy_level,
                shift_template_id, is_available, is_generalist
            FROM operators
            ORDER BY id
        """)
        return self.cursor.fetchall()

    def get_operator_by_id(self, operator_id):
        """Получить оператора по ID"""
        self.cursor.execute("""
            SELECT 
                id, full_name, gender, birth_date, start_date,
                patience_level, stress_resistance_level, empathy_level,
                shift_template_id, is_available, is_generalist
            FROM operators 
            WHERE id = %s
        """, (operator_id,))
        return self.cursor.fetchone()

    def get_available_operators(self):
        """Получить только доступных операторов"""
        self.cursor.execute("""
            SELECT 
                id, full_name, gender, shift_template_id, is_generalist
            FROM operators
            WHERE is_available = TRUE
            ORDER BY id
        """)
        return self.cursor.fetchall()

    def get_generalists(self, only_available=True):
        """
        Получить операторов-универсалов.

        Параметры:
            only_available (bool): если True — только доступные
        """
        query = """
            SELECT 
                id, full_name, gender,
                patience_level, stress_resistance_level, empathy_level,
                shift_template_id, is_available
            FROM operators
            WHERE is_generalist = TRUE
        """
        if only_available:
            query += " AND is_available = TRUE"
        query += " ORDER BY id"

        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_available_generalists(self):
        """Получить доступных универсалов"""
        query = """
            SELECT 
                id, 
                full_name,
                gender,
                patience_level,
                stress_resistance_level,
                empathy_level,
                shift_template_id
            FROM operators
            WHERE is_available = TRUE 
                AND is_generalist = TRUE
            ORDER BY patience_level DESC, stress_resistance_level DESC
        """
        self.cursor.execute(query)
        return self.cursor.fetchall()

    def get_available_operators_by_intent(self, intent_code):
        """
        Получить доступных операторов, у которых есть указанная специализация
        """
        query = """
            SELECT 
                o.id, 
                o.full_name,
                o.gender,
                o.patience_level,
                o.stress_resistance_level,
                o.empathy_level,
                os.proficiency_level,
                os.is_primary,
                o.shift_template_id
            FROM operators o
            JOIN operator_specializations os ON o.id = os.operator_id
            JOIN specializations s ON os.specialization_id = s.id
            WHERE o.is_available = TRUE 
                AND s.intent_code = %s
            ORDER BY os.is_primary DESC, os.proficiency_level DESC, o.id
        """
        self.cursor.execute(query, (intent_code,))
        return self.cursor.fetchall()

    def get_available_on_shift_operators_by_intent(self, intent_code):
        current_time = datetime.now().time()

        query = """
            SELECT 
                o.id, 
                o.full_name,
                o.gender,
                o.birth_date,
                o.start_date,
                o.patience_level,
                o.stress_resistance_level,
                o.empathy_level,
                os.proficiency_level,
                os.is_primary,
                o.shift_template_id,
                st.start_time,
                st.end_time,
                s.name as specialization_name
            FROM operators o
            JOIN operator_specializations os ON o.id = os.operator_id
            JOIN specializations s ON os.specialization_id = s.id
            JOIN shift_templates st ON o.shift_template_id = st.id
            WHERE o.is_available = TRUE 
                AND s.intent_code = %s
                AND (
                    -- обычные смены (start <= end)
                    (st.start_time <= st.end_time AND st.start_time <= %s AND st.end_time >= %s)
                    OR
                    -- смены через полночь (start > end)
                    (st.start_time > st.end_time AND (%s >= st.start_time OR %s <= st.end_time))
                )
            ORDER BY os.is_primary DESC, os.proficiency_level DESC, o.id
        """
        self.cursor.execute(query, (intent_code, current_time, current_time, current_time, current_time))
        return self.cursor.fetchall()

    def get_available_on_shift_generalists(self):
        current_time = datetime.now().time()

        query = """
            SELECT 
                o.id, 
                o.full_name,
                o.gender,
                o.birth_date,
                o.start_date,
                o.patience_level,
                o.stress_resistance_level,
                o.empathy_level,
                NULL as proficiency_level,
                FALSE as is_primary,
                o.shift_template_id,
                st.start_time,
                st.end_time,
                NULL as specialization_name
            FROM operators o
            JOIN shift_templates st ON o.shift_template_id = st.id
            WHERE o.is_available = TRUE 
                AND o.is_generalist = TRUE
                AND (
                    -- обычные смены (start <= end)
                    (st.start_time <= st.end_time AND st.start_time <= %s AND st.end_time >= %s)
                    OR
                    -- смены через полночь (start > end)
                    (st.start_time > st.end_time AND (%s >= st.start_time OR %s <= st.end_time))
                )
            ORDER BY o.patience_level DESC, o.stress_resistance_level DESC
        """
        self.cursor.execute(query, (current_time, current_time, current_time, current_time))
        return self.cursor.fetchall()

    def add_operator(self, full_name, gender, birth_date, start_date,
                     patience, stress, empathy, shift_id,
                     is_available=True, is_generalist=False):
        """Добавить оператора"""
        self.cursor.execute("""
            INSERT INTO operators 
            (full_name, gender, birth_date, start_date, patience_level,
             stress_resistance_level, empathy_level, shift_template_id, 
             is_available, is_generalist)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (full_name, gender, birth_date, start_date,
              patience, stress, empathy, shift_id, is_available, is_generalist))
        self.conn.commit()
        return self.cursor.fetchone()[0]

    def update_availability(self, operator_id, is_available):
        """Обновить статус доступности оператора"""
        self.cursor.execute("""
            UPDATE operators 
            SET is_available = %s 
            WHERE id = %s
        """, (is_available, operator_id))
        self.conn.commit()
        return self.cursor.rowcount

    def update_generalist(self, operator_id, is_generalist):
        """Обновить статус универсала"""
        self.cursor.execute("""
            UPDATE operators 
            SET is_generalist = %s 
            WHERE id = %s
        """, (is_generalist, operator_id))
        self.conn.commit()
        return self.cursor.rowcount

    def update_operator(self, operator_id, full_name, gender, birth_date, start_date,
                        patience, stress, empathy, shift_id, is_available, is_generalist):
        """Обновить все поля оператора (кроме специализаций)"""
        self.cursor.execute("""
            UPDATE operators 
            SET full_name = %s,
                gender = %s,
                birth_date = %s,
                start_date = %s,
                patience_level = %s,
                stress_resistance_level = %s,
                empathy_level = %s,
                shift_template_id = %s,
                is_available = %s,
                is_generalist = %s
            WHERE id = %s
        """, (full_name, gender, birth_date, start_date,
              patience, stress, empathy, shift_id, is_available, is_generalist,
              operator_id))
        self.conn.commit()
        return self.cursor.rowcount

    def update_operator_specialization(self, operator_id, specialization_id, proficiency_level, is_primary):
        """Обновить специализацию оператора (если есть) или добавить новую"""
        # проверка, есть ли уже такая специализация
        self.cursor.execute("""
            SELECT id FROM operator_specializations 
            WHERE operator_id = %s AND specialization_id = %s
        """, (operator_id, specialization_id))
        exists = self.cursor.fetchone()

        if exists:
            # обновление существующей
            self.cursor.execute("""
                UPDATE operator_specializations 
                SET proficiency_level = %s, is_primary = %s
                WHERE operator_id = %s AND specialization_id = %s
            """, (proficiency_level, is_primary, operator_id, specialization_id))
        else:
            # добавление новой
            self.cursor.execute("""
                INSERT INTO operator_specializations (operator_id, specialization_id, proficiency_level, is_primary)
                VALUES (%s, %s, %s, %s)
            """, (operator_id, specialization_id, proficiency_level, is_primary))

        self.conn.commit()
        return True

    # Специализации

    def get_all_specializations(self):
        """Получить все специализации"""
        self.cursor.execute("""
            SELECT id, name, intent_code 
            FROM specializations 
            ORDER BY id
        """)
        return self.cursor.fetchall()

    def get_operator_specializations(self, operator_id):
        """Получить специализации оператора"""
        self.cursor.execute("""
            SELECT 
                s.id,
                s.name,
                os.proficiency_level,
                os.is_primary
            FROM operator_specializations os
            JOIN specializations s ON os.specialization_id = s.id
            WHERE os.operator_id = %s
            ORDER BY os.is_primary DESC, os.proficiency_level DESC
        """, (operator_id,))
        return self.cursor.fetchall()

    # Смены

    def get_all_shifts(self):
        """Получить все шаблоны смен"""
        self.cursor.execute("""
            SELECT id, shift_name, start_time, end_time 
            FROM shift_templates 
            ORDER BY id
        """)
        return self.cursor.fetchall()

    def get_operators_by_shift(self, shift_id):
        """Получить операторов по ID смены"""
        self.cursor.execute("""
            SELECT o.id, o.full_name, o.gender, o.is_available, o.is_generalist
            FROM operators o
            WHERE o.shift_template_id = %s
            ORDER BY o.id
        """, (shift_id,))
        return self.cursor.fetchall()

    def get_shift_by_id(self, shift_id):
        """Получить данные смены по ID"""
        query = """
            SELECT id, shift_name, start_time, end_time
            FROM shift_templates
            WHERE id = %s
        """
        self.cursor.execute(query, (shift_id,))
        row = self.cursor.fetchone()
        if row:
            return {
                'id': row[0],
                'name': row[1],
                'start_time': row[2],
                'end_time': row[3]
            }
        return None

    # Статистика

    def get_stats(self):
        """Общая статистика"""
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_available THEN 1 ELSE 0 END) as available,
                SUM(CASE WHEN is_generalist THEN 1 ELSE 0 END) as generalists,
                SUM(CASE WHEN gender = 'M' THEN 1 ELSE 0 END) as males,
                SUM(CASE WHEN gender = 'F' THEN 1 ELSE 0 END) as females,
                ROUND(AVG(patience_level)::numeric, 2) as avg_patience,
                ROUND(AVG(stress_resistance_level)::numeric, 2) as avg_stress,
                ROUND(AVG(empathy_level)::numeric, 2) as avg_empathy
            FROM operators
        """)
        return self.cursor.fetchone()
