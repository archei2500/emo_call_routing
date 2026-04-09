import psycopg2
from db_config import DB_CONFIG


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

    # ========== ОПЕРАТОРЫ ==========

    def get_all_operators(self):
        """Получить всех операторов"""
        self.cursor.execute("""
            SELECT 
                id, full_name, birth_date, start_date,
                patience_level, stress_resistance_level, empathy_level,
                shift_template_id, is_available
            FROM operators
            ORDER BY id
        """)
        return self.cursor.fetchall()

    def get_operator_by_id(self, operator_id):
        """Получить оператора по ID"""
        self.cursor.execute("""
            SELECT 
                id, full_name, birth_date, start_date,
                patience_level, stress_resistance_level, empathy_level,
                shift_template_id, is_available
            FROM operators 
            WHERE id = %s
        """, (operator_id,))
        return self.cursor.fetchone()

    def get_available_operators(self):
        """Получить только доступных операторов"""
        self.cursor.execute("""
            SELECT 
                id, full_name, shift_template_id
            FROM operators
            WHERE is_available = TRUE
            ORDER BY id
        """)
        return self.cursor.fetchall()

    def add_operator(self, full_name, birth_date, start_date,
                     patience, stress, empathy, shift_id, is_available=True):
        """Добавить оператора"""
        self.cursor.execute("""
            INSERT INTO operators 
            (full_name, birth_date, start_date, patience_level,
             stress_resistance_level, empathy_level, shift_template_id, is_available)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (full_name, birth_date, start_date,
              patience, stress, empathy, shift_id, is_available))
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

    # ========== СПЕЦИАЛИЗАЦИИ ==========

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

    # ========== СМЕНЫ ==========

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
            SELECT o.id, o.full_name, o.is_available
            FROM operators o
            WHERE o.shift_template_id = %s
            ORDER BY o.id
        """, (shift_id,))
        return self.cursor.fetchall()

    # ========== СТАТИСТИКА ==========

    def get_stats(self):
        """Общая статистика"""
        self.cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN is_available THEN 1 ELSE 0 END) as available,
                ROUND(AVG(patience_level)::numeric, 2) as avg_patience,
                ROUND(AVG(stress_resistance_level)::numeric, 2) as avg_stress,
                ROUND(AVG(empathy_level)::numeric, 2) as avg_empathy
            FROM operators
        """)
        return self.cursor.fetchone()

# if __name__ == "__main__":
#     with ContactCenterDB() as db:
#         # Все операторы
#         operators = db.get_all_operators()
#         print(f"📋 Всего операторов: {len(operators)}")
#
#         # Статистика
#         stats = db.get_stats()
#         print(f"\n📊 Статистика:")
#         print(f"  Доступно: {stats[1]} из {stats[0]}")
#         print(f"  Средняя терпеливость: {stats[2]}")
#         print(f"  Средняя стрессоустойчивость: {stats[3]}")
#         print(f"  Средняя эмпатия: {stats[4]}")
#
#         # Все специализации
#         specs = db.get_all_specializations()
#         print(f"\n📌 Специализации:")
#         for spec in specs:
#             print(f"  {spec[0]}. {spec[1]}")
#
#         # Все смены
#         shifts = db.get_all_shifts()
#         print(f"\n🕐 Смены:")
#         for shift in shifts:
#             print(f"  {shift[0]}. {shift[1]} ({shift[2]} - {shift[3]})")
