from database import ContactCenterDB
from datetime import date, datetime
import math

# пороги valence-arousal-dominance для подозрения в агрессии или дистрессе
TRIGGER_ANGER = T_VAD_ANGER = 0.08
TRIGGER_DISTRESS = T_VAD_DISTRESS = 0.15
# индивидуальные пороги моделей
T_GIGA_ANGER = 0.39
T_GIGA_DISTRESS = 0.50
T_TEXT_ANGER = 0.31
T_TEXT_DISTRESS = 0.70

# базовые пороги
BASE_T_ANGER = 0.293
BASE_T_DISTRESS = 0.493

birth_date_idx = 3
start_date_idx = 4
patience_idx = 5
stress_idx = 6
empathy_idx = 7
proficiency_idx = 8
is_primary_idx = 9
shift_idx = 10
start_time_idx = 11
end_time_idx = 12
gender_idx = 2
specialization_name_idx = 13


def calculate_age(birth_date):
    """
    Вычисляет возраст по дате рождения
    """
    if not birth_date:
        return 35  # значение по умолчанию
    today = date.today()
    age = today.year - birth_date.year
    # был ли день рождения в этом году
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1

    return age


def calculate_experience_years(start_date):
    """Стаж работы в годах"""
    if not start_date:
        return 0

    today = datetime.now().date()
    years = today.year - start_date.year

    if (today.month, today.day) < (start_date.month, start_date.day):
        years -= 1

    return years


def get_shift_fatigue(shift_start_time, shift_end_time):
    """
    Возвращает коэффициент усталости (0 = только начал, 1 = конец смены)
    Поддерживает смены, переходящие через полночь (например, 16:00 - 00:00)
    """
    now = datetime.now().time()

    start_min = shift_start_time.hour * 60 + shift_start_time.minute
    end_min = shift_end_time.hour * 60 + shift_end_time.minute
    now_min = now.hour * 60 + now.minute

    # Смена через полночь (например, 16:00 - 00:00)
    if start_min > end_min:
        # Проверяем, находимся ли мы в смене
        if now_min >= start_min or now_min <= end_min:
            # сколько всего минут в смене
            total_minutes = (24 * 60 - start_min) + end_min

            # сколько минут уже отработал
            if now_min >= start_min:
                worked_minutes = now_min - start_min
            else:
                worked_minutes = (24 * 60 - start_min) + now_min

            if total_minutes == 0:
                return 0.0

            fatigue = worked_minutes / total_minutes
            if fatigue < 0:
                return 0.0
            if fatigue > 1:
                return 1.0
            return fatigue
        else:
            return 1.0

    # Обычная смена (start <= end)
    else:
        if now_min < start_min or now_min > end_min:
            return 1.0

        total_minutes = end_min - start_min

        if total_minutes == 0:
            return 0.0

        worked_minutes = now_min - start_min
        fatigue = worked_minutes / total_minutes

        if fatigue < 0:
            return 0.0
        if fatigue > 1:
            return 1.0
        return fatigue


def filter_by_effective_skill(operators, skill_idx, skill_name):
    """
    Фильтрует операторов по эффективному навыку с динамическим порогом,
    затем сортирует по опыту.

    Args:
        operators: список операторов (кортежи)
        skill_idx: индекс базового навыка в кортеже
        skill_name: название навыка для reason ("стрессоустойчивость" или "эмпатия")

    Returns:
        tuple: (отфильтрованный_список, reason_строка)
    """
    if not operators:
        return operators, ""

    operators_with_effective = []
    for op in operators:
        base_skill = op[skill_idx]
        start_time = op[start_time_idx]
        end_time = op[end_time_idx]
        fatigue = get_shift_fatigue(start_time, end_time)
        effective_skill = base_skill * (1 - fatigue)
        operators_with_effective.append((op, effective_skill))

    # максимум и порог (округление вниз)
    max_effective = max(operators_with_effective, key=lambda x: x[1])[1]
    threshold = math.floor(max_effective)
    # оставляем тех, чей навык >= порога
    good_operators = [(op, eff) for op, eff in operators_with_effective if eff >= threshold]
    # сортировка по опыту (от большего к меньшему)
    good_operators.sort(
        key=lambda x: calculate_experience_years(x[0][start_date_idx]),
        reverse=True
    )
    filtered_operators = [op for op, eff in good_operators]
    top_exp = calculate_experience_years(filtered_operators[0][start_date_idx])
    reason = (f" → Эфф. {skill_name}: максимум {max_effective:.2f}, "
              f"порог {threshold}, выбрано {len(filtered_operators)} чел, "
              f"лучший по опыту ({top_exp} лет)")

    return filtered_operators, reason


def filter_by_effective_patience(operators, operators_with_age):
    """
    Фильтрует операторов по эффективному терпению с динамическим порогом,
    затем сортирует по опыту. Дополнительно учитывает возраст 50+ для пожилых клиентов.

    Returns:
        tuple: (отфильтрованный_список, reason_строка)
    """
    if not operators:
        return operators, ""

    operators_with_effective = []
    for op, age in operators_with_age:
        base_patience = op[patience_idx]
        start_time = op[start_time_idx]
        end_time = op[end_time_idx]
        fatigue = get_shift_fatigue(start_time, end_time)
        effective_patience = base_patience * (1 - fatigue)
        operators_with_effective.append((op, age, effective_patience))

    max_effective = max(eff for _, _, eff in operators_with_effective)
    threshold = math.floor(max_effective)

    good_operators = [(op, age, eff) for op, age, eff in operators_with_effective if eff >= threshold]
    senior_ops = [(op, age, eff) for op, age, eff in good_operators if age >= 50]
    if senior_ops:
        # есть 50+ - сортируем их по опыту
        senior_ops.sort(key=lambda x: calculate_experience_years(x[0][start_date_idx]), reverse=True)
        filtered_operators = [op for op, age, eff in senior_ops]
        top_exp = calculate_experience_years(filtered_operators[0][start_date_idx])
        reason = (f" → Клиент 60+: эфф. терпение макс {max_effective:.2f}, порог {threshold}, "
                  f"возраст 50+, лучший по опыту ({top_exp} лет)")
    else:
        # нет 50+ - сортируем всех прошедших порог по опыту
        good_operators.sort(key=lambda x: calculate_experience_years(x[0][start_date_idx]), reverse=True)
        filtered_operators = [op for op, age, eff in good_operators]
        top_exp = calculate_experience_years(filtered_operators[0][start_date_idx])
        reason = (f" → Клиент 60+: эфф. терпение макс {max_effective:.2f}, порог {threshold}, "
                  f"нет операторов 50+, лучший по опыту ({top_exp} лет)")

    return filtered_operators, reason


def extract_demographic_features(paralinguistic_features):
    """
    Извлекает демографические признаки (пол и возрастную категорию) из паралингвистических признаков.
    Возвращает кортеж (gender, client_category).
    """
    gender = paralinguistic_features.get("ag_result", {}).get("gender", {}).get("predicted", None)
    age = paralinguistic_features.get("ag_result", {}).get("age", {}).get("years", 0)
    if 14 <= age <= 29:
        age_category = "young"  # молодые операторы (18-29)
    elif 30 <= age <= 49:
        age_category = "adult"  # к взрослым (30-49)
    elif age >= 50:
        age_category = "senior"  # к 50+
    else:
        age_category = "unknown"

    return gender, age, age_category


def determine_emotion_route(paralinguistic_features, semantic_emotion, routing_mode):
    """
    Определяет маршрут на основе эмоционального состояния клиента.
    Использует VAD-модель, GigaAM (если доступна) и семантическую эмоцию.
    Возвращает строку: "stress_operator", "empathy_operator" или "standard_operator".
    """
    if routing_mode in [0, 2]:
        # получение рисков (предсказаны VAD-моделью)
        vad_risks = paralinguistic_features.get("emo_vad_result", {}).get("risks", {})
        anger_risk = vad_risks.get("anger", 0.0)
        distress_risk = vad_risks.get("distress", 0.0)
        # получение предсказаний GigaAM, если они есть
        giga_data = paralinguistic_features.get("emo_result", None)
        giga_probs = giga_data.get("probs", None) if giga_data else None
        g_anger = giga_probs.get("angry", 0.0) if giga_probs else 0.0
        g_distress = giga_probs.get("sad", 0.0) if giga_probs else 0.0
        # текстовые предсказания
        text_anger = semantic_emotion.get("probabilities", {}).get("anger", 0.0) if semantic_emotion else 0.0
        text_distress = (semantic_emotion.get("probabilities", {}).get("sadness", 0.0) +
                         semantic_emotion.get("probabilities", {}).get("fear", 0.0)) if semantic_emotion else 0.0

        has_giga = bool(giga_data)
        has_text = bool(semantic_emotion)
        # оценка уверенности по индивидуальным порогам
        vad_c_a = anger_risk >= T_VAD_ANGER
        giga_c_a = g_anger >= T_GIGA_ANGER
        text_c_a = text_anger >= T_TEXT_ANGER
        vad_c_d = distress_risk >= T_VAD_DISTRESS
        giga_c_d = g_distress >= T_GIGA_DISTRESS
        text_c_d = text_distress >= T_TEXT_DISTRESS

        audio_strong = (vad_c_a and giga_c_a) or (vad_c_d and giga_c_d)
        text_strong = text_c_a or text_c_d

        # динамические веса и пороги
        w = {"vad": 0.0, "giga": 0.0, "text": 0.0}
        t_anger = BASE_T_ANGER
        t_distress = BASE_T_DISTRESS
        scenario_handled = False

        if has_giga and has_text:
            scenario_handled = True
            if audio_strong:
                # Аудио уверено - верим аудио, текст приглушаем, пороги снижаем
                w = {"vad": 0.25, "giga": 0.60, "text": 0.15}
                t_anger, t_distress = BASE_T_ANGER * 0.7, BASE_T_DISTRESS * 0.7
            elif text_strong:
                # Текст уверен, аудио нет - верим тексту, пороги повышаем
                w = {"vad": 0.15, "giga": 0.25, "text": 0.60}
                t_anger, t_distress = BASE_T_ANGER * 1.3, BASE_T_DISTRESS * 1.3
            else:
                # Нейтральный фон → стандартный баланс
                if anger_risk >= distress_risk:
                    w = {"vad": 0.25, "giga": 0.50, "text": 0.25}
                else:
                    w = {"vad": 0.25, "giga": 0.35, "text": 0.40}

        elif has_giga:
            scenario_handled = True
            w = {"vad": 0.30, "giga": 0.70, "text": 0.0}
            if audio_strong:
                t_anger, t_distress = BASE_T_ANGER * 0.8, BASE_T_DISTRESS * 0.8

        elif has_text:
            scenario_handled = True
            vad_triggered = vad_c_a or vad_c_d
            if vad_triggered:
                # VAD видит эмоцию, но GigaAM не успела - верим VAD больше
                w = {"vad": 0.50, "giga": 0.0, "text": 0.50}
            else:
                # VAD спокойна - верим только тексту
                w = {"vad": 0.10, "giga": 0.0, "text": 0.90}
                t_anger, t_distress = BASE_T_ANGER * 1.2, BASE_T_DISTRESS * 1.2

        # итоговый расчёт и решение
        if scenario_handled:
            final_anger = (w["vad"] * anger_risk +
                           w["giga"] * g_anger +
                           w["text"] * text_anger)
            final_distress = (w["vad"] * distress_risk +
                              w["giga"] * g_distress +
                              w["text"] * text_distress)
            print("Предсказания по семантике:", text_anger, text_distress)
            print("Финальные риски:", "Гнев - ", final_anger, "Дистресс - ", final_distress)

            if final_anger > t_anger:
                emo_route = "stress_operator"
            elif final_distress > t_distress:
                emo_route = "empathy_operator"
            else:
                emo_route = "standard_operator"
        else:
            emo_route = "standard_operator"
    elif semantic_emotion:
        predicted_emotion = semantic_emotion.get("primary_emotion")
        if predicted_emotion == "anger":
            emo_route = "stress_operator"
        elif predicted_emotion in ["sadness", "fear"]:
            emo_route = "empathy_operator"
        else:
            emo_route = "standard_operator"
    else:
        emo_route = "standard_operator"

    return emo_route


def route_inquiry(paralinguistic_features, intent=None, semantic_emotion=None, routing_mode=0):
    # Подготовка
    # извлечение демографических признаков
    client_gender = None
    client_age = 0
    client_category = "unknown"
    if routing_mode in [0, 1]:
        client_gender, client_age, client_category = extract_demographic_features(paralinguistic_features)

    # определение того, клиент с каким качеством нужен (стрессоустойчивость/эмпатия)
    emo_route = determine_emotion_route(paralinguistic_features, semantic_emotion, routing_mode)

    # тема обращения
    intent_name = None
    if intent:
        intent_name = intent.get("predicted_intent")
        intent_conf = intent.get("confidence", 0.0)
        is_uncertain = (intent_conf < 0.5)
    else:
        is_uncertain = True

    # маршрутизация
    suitable_operators = []
    # получаем БД операторов
    with ContactCenterDB() as db:
        # 1. Сначала по теме обращения
        if not is_uncertain:
            specialists = db.get_available_on_shift_operators_by_intent(intent_name)

            if specialists:
                primary_specialists = [s for s in specialists if s[is_primary_idx]]
                secondary_specialists = [s for s in specialists if not s[is_primary_idx]]

                if primary_specialists:
                    # сортировка primary по уровню владения (proficiency_level) по убыванию
                    primary_specialists.sort(key=lambda x: x[proficiency_idx], reverse=True)
                    suitable_operators = primary_specialists
                    reason = f"Тема '{intent_name}' → Найдено {len(primary_specialists)} основных специалистов"
                else:
                    secondary_specialists.sort(key=lambda x: x[proficiency_idx], reverse=True)
                    suitable_operators = secondary_specialists
                    reason = f"Тема '{intent_name}' → Нет основных специалистов, найдено {len(secondary_specialists)} из остальных"
            else:
                generalists = db.get_available_on_shift_generalists()
                if generalists:
                    suitable_operators = generalists
                    reason = f"Нет доступных специалистов для '{intent_name}' → Но есть {len(generalists)} универсальных операторов"
                else:
                    suitable_operators = []
                    reason = f"Нет специалистов для '{intent_name}', и универсальные недоступны!"
        else:
            generalists = db.get_available_on_shift_generalists()
            if generalists:
                suitable_operators = generalists
                reason = f"Подобрано {len(generalists)} универсальных специалистов"
            else:
                suitable_operators = []
                reason = "Нет доступных универсальных специалистов"

        if suitable_operators:
            # 2. Далее по эмоциональному состоянию
            if emo_route == "stress_operator":
                filtered_operators, filter_reason = filter_by_effective_skill(
                    operators=suitable_operators,
                    skill_idx=stress_idx,
                    skill_name="стрессоустойчивость"
                )
                if filtered_operators:  # если кто-то остался после фильтрации
                    suitable_operators = filtered_operators
                    reason += filter_reason
                else:
                    reason += f" → Фильтр по стрессоустойчивости отсеял всех, оставлен предыдущий список"
            elif emo_route == "empathy_operator":
                filtered_operators, filter_reason = filter_by_effective_skill(
                    operators=suitable_operators,
                    skill_idx=empathy_idx,
                    skill_name="эмпатия"
                )
                if filtered_operators:
                    suitable_operators = filtered_operators
                    reason += filter_reason
                else:
                    reason += f" → Фильтр по эмпатии отсеял всех, оставлен предыдущий список"
            elif emo_route == "standard_operator":
                reason += f" → Стандартный, без эмоционального фильтра"

            # 3. По демографии, если есть такие признаки
            if routing_mode in [0, 1]:
                # 3.1. По возрасту
                # вычисляем возраст для каждого оператора
                operators_with_age = []
                for op in suitable_operators:
                    birth_date = op[birth_date_idx]
                    age = calculate_age(birth_date)
                    operators_with_age.append((op, age))

                # правило для пожилых клиентов 60+
                if client_age >= 60:
                    filtered_operators, filter_reason = filter_by_effective_patience(
                        operators=suitable_operators,
                        operators_with_age=operators_with_age
                    )
                    if filtered_operators:
                        suitable_operators = filtered_operators
                        reason += filter_reason
                    else:
                        reason += f" → Фильтр для клиентов 60+ отсеял всех, оставлен предыдущий список"
                # обычная фильтрация по возрастной группе (для всех остальных)
                else:
                    if client_category != "unknown":
                        if client_category == "young":
                            target_min, target_max = 18, 29
                        elif client_category == "adult":
                            target_min, target_max = 30, 49
                        elif client_category == "senior":
                            target_min, target_max = 50, 99
                        else:
                            target_min, target_max = None, None

                        if target_min is not None and target_max is not None:
                            in_range = [(op, age) for op, age in operators_with_age
                                        if target_min <= age <= target_max]
                            if in_range:
                                suitable_operators = [op for op, age in in_range]
                                reason += f" → Возраст {target_min}-{target_max} лет ({client_category})"
                            else:
                                reason += f" → Нет операторов {target_min}-{target_max} лет, без фильтрации по возрасту"
                    else:
                        reason += f" → Возрастная категория неизвестна, без фильтрации"

                # 3.2. Выбор по полу
                if len(suitable_operators) > 1:
                    if client_gender in ["male", "female"]:
                        op_gender = 'female' if op[gender_idx] == 'F' else 'male' if op[gender_idx] == 'M' else None
                        same_gender = [op for op in suitable_operators if op_gender == client_gender]

                        if same_gender:
                            suitable_operators = same_gender
                            reason += f" → Пол: {client_gender}"
                        else:
                            reason += f" → Нет операторов пола '{client_gender}'"

            best_op = suitable_operators[0]
            age = calculate_age(best_op[birth_date_idx])
            experience_years = calculate_experience_years(best_op[start_date_idx])
            pos = best_op[specialization_name_idx] or "Универсальный специалист"
            start_time_str = best_op[start_time_idx].strftime("%H:%M")
            end_time_str = best_op[end_time_idx].strftime("%H:%M")
            result = (
                    f"ID: {best_op[0]}\n"
                    f"ФИО: {best_op[1]}\n"
                    f"Должность (подходящая или универсальная): {pos}\n"
                    f"Пол: {best_op[gender_idx]}\n"
                    f"Возраст: {age} лет\n"
                    f"Опыт: {experience_years} лет\n"
                    f"Терпение: {best_op[patience_idx]}/5\n"
                    f"Стрессоустойчивость: {best_op[stress_idx]}/5\n"
                    f"Эмпатия: {best_op[empathy_idx]}/5\n"
                    + (f"Профессиональный уровень: {best_op[proficiency_idx]}/5\n"
                       f"Основная специализация: {'да' if best_op[is_primary_idx] else 'нет'}\n"
                       if best_op[proficiency_idx] is not None else "") +
                    f"Смена: {start_time_str} - {end_time_str}"
            )
            # отладка
            print(f"[ROUTING] {reason}")
            return result
        else:
            print(f"[ROUTING] {reason}")
            return "Извините, в данный момент нет доступных операторов. Пожалуйста, позвоните позже или оставьте заявку на сайте."
