from database import ContactCenterDB
from datetime import date, datetime
import math

# пороги (нужно подобрать экспериментально или эмпирически)
T_anger = 0.3
T_distress = 0.25

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
    """
    now = datetime.now().time()
    # если смена ещё не началась или уже закончилась
    if now < shift_start_time or now > shift_end_time:
        return 1.0
    # сколько всего минут в смене
    total_minutes = (shift_end_time.hour * 60 + shift_end_time.minute) - \
                    (shift_start_time.hour * 60 + shift_start_time.minute)
    # сколько минут уже отработал
    worked_minutes = (now.hour * 60 + now.minute) - \
                     (shift_start_time.hour * 60 + shift_start_time.minute)
    return worked_minutes / total_minutes


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


def route_inquiry(paralinguistic_features, intent=None, semantic_emotion=None, routing_mode=0):
    # Что у нас вообще есть?
    # Пол
    # Возраст
    # Эмоция - по голосу и по тексту
    # Интент
    # демографические признаки
    gender = None
    client_category = "unknown"
    client_age = 0
    if routing_mode in [0, 1]:
        gender = paralinguistic_features["ag_result"]["gender"]["predicted"]
        client_age = paralinguistic_features["ag_result"]["age"]["years"]
        if 14 <= client_age <= 29:
            client_category = "young"  # молодые операторы (18-29)
        elif 30 <= client_age <= 49:
            client_category = "adult"  # к взрослым (30-49)
        elif client_age >= 50:
            client_category = "senior"  # к 50+
        else:
            client_category = "unknown"

    # эмоциональное состояние
    if routing_mode in [0, 2]:
        # получение рисков (предсказаны VAD-моделью)
        vad_risks = paralinguistic_features.get("emo_vad_result", {}).get("risks", {})
        anger_risk = vad_risks.get("anger", 0.0)
        distress_risk = vad_risks.get("distress", 0.0)
        # получение предсказаний GigaAM, если они есть
        giga_data = paralinguistic_features.get("emo_result")
        giga_probs = giga_data.get("probs", None)
        g_anger = giga_probs.get("angry", 0.0) if giga_probs else 0.0
        g_distress = giga_probs.get("sad", 0.0) if giga_probs else 0.0
        # текстовые предсказания
        t_anger = semantic_emotion.get("probabilities", {}).get("anger", 0.0) if semantic_emotion else 0.0
        t_distress = (semantic_emotion.get("probabilities", {}).get("sadness", 0.0) +
                      semantic_emotion.get("probabilities", {}).get("fear", 0.0)) if semantic_emotion else 0.0
        if anger_risk >= distress_risk:
            w_vad, w_giga, w_text = 0.25, 0.50, 0.25
        else:
            w_vad, w_giga, w_text = 0.25, 0.35, 0.40
        # нормализация весов под доступные модели
        active_weights = [w_vad]
        if giga_data:
            active_weights.append(w_giga)
        if semantic_emotion:
            active_weights.append(w_text)
        sum_w = sum(active_weights)
        w_vad_n = w_vad / sum_w
        w_giga_n = w_giga / sum_w if giga_data else 0.0
        w_text_n = w_text / sum_w if semantic_emotion else 0.0
        # итоговые риски
        final_anger = (w_vad_n * anger_risk + w_giga_n * g_anger + w_text_n * t_anger)
        final_distress = (w_vad_n * distress_risk + w_giga_n * g_distress + w_text_n * t_distress)
        # выбор качества оператора по эмоциональному состоянию
        if final_anger > T_anger:
            emo_route = "stress_operator"
        elif final_distress > T_distress:
            emo_route = "empathy_operator"
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
                    patience = op[patience_idx]
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
                    client_gender = paralinguistic_features.get("ag_result", {}).get("gender", {}).get("predicted")
                    if client_gender in ["male", "female"]:
                        same_gender = [op for op in suitable_operators if op[gender_idx] == client_gender]

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


