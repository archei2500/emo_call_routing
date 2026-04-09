# пороги (нужно подобрать экспериментально или эмпирически)
T_anger = 0.3
T_distress = 0.25


def route_inquiry(paralinguistic_features, intent=None, semantic_emotion=None, routing_mode=0):
    # Что у нас вообще есть?
    # Пол
    # Возраст
    # Эмоция - по голосу и по тексту
    # Интент
    # демографические признаки
    if routing_mode in [0, 1]:
        gender = paralinguistic_features["ag_result"]["gender"]["predicted"]
        age = paralinguistic_features["ag_result"]["age"]["years"]
        if 14 <= age <= 29:
            client_category = "young"  # молодые операторы (18-29)
        elif 30 <= age <= 49:
            client_category = "adult"  # к взрослым (30-49)
        elif age >= 50:
            client_category = "senior"  # к 50+

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



