import json
import re
import random
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent

with open(ROOT / "intents.json", encoding="utf-8") as f:
    DATA = json.load(f)

with open(ROOT / "destinations.json", encoding="utf-8") as f:
    PLACES = json.load(f)


# Навчальний датасет
texts = []
labels = []

for intent, phrases in DATA.items():
    for phrase in phrases:
        texts.append(phrase)
        labels.append(intent)


# Модель машинного навчання
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        lowercase=True
    )),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

model.fit(texts, labels)


# Відмінки назв міст
DEST_ALIASES = {
    "львів": [
        "львів", "львова", "львові", "львовом"
    ],
    "київ": [
        "київ", "києва", "києві", "києвом"
    ],
    "одеса": [
        "одеса", "одеси", "одесі", "одесу"
    ],
    "карпати": [
        "карпати", "карпатах", "карпат",
        "карпатами"
    ],
    "яремче": [
        "яремче"
    ],
    "кам’янець-подільський": [
        "кам’янець-подільський",
        "кам'янець-подільський",
        "кам’янця-подільського",
        "кам'янця-подільського",
        "кам’янці-подільському",
        "кам'янці-подільському"
    ]
}


# Виділення сутностей
def extract_entities(text):
    low = text.lower()
    entities = {}

    for destination, aliases in DEST_ALIASES.items():
        if any(alias in low for alias in aliases):
            entities["destination"] = destination
            break

    money = re.search(
        r"\b(\d[\d\s]{0,7})\s*"
        r"(?:грн|грив(?:ень|ні|ню|нями)?)\b",
        low
    )

    if money:
        entities["budget"] = int(
            money.group(1).replace(" ", "")
        )

    days = re.search(
        r"\b(\d+)\s*"
        r"(?:день|дні|днів|дня|добу|доби|діб)\b",
        low
    )

    if days:
        entities["duration"] = int(days.group(1))

    # Тривалість словами
    word_days = {
        "один день": 1,
        "одного дня": 1,
        "два дні": 2,
        "двох днів": 2,
        "три дні": 3,
        "трьох днів": 3,
        "чотири дні": 4,
        "п'ять днів": 5,
        "п’ять днів": 5
    }

    for phrase, number in word_days.items():
        if phrase in low:
            entities["duration"] = number
            break

    seasons = {
        "весна": ["весн", "навесні"],
        "літо": ["літ", "влітку"],
        "осінь": ["осін", "восени"],
        "зима": ["зим", "взимку"]
    }

    for season, variants in seasons.items():
        if any(v in low for v in variants):
            entities["season"] = season
            break

    return entities

def detect_intent_by_keywords(message):
    low = message.lower()

    keywords = {
        "activities": [
            "розваг", "розваж", "активност",
            "дозвілл", "чим зайнятися",
            "що робити на відпочинку"
        ],
        "food": [
            "скуштувати", "поїсти",
            "страв", "кухн", "їжа",
            "делікатес"
        ],
        "itinerary": [
            "склади маршрут", "скласти маршрут",
            "план подорожі", "розплануй",
            "маршрут на", "план екскурсій"
        ],
        "attractions": [
            "що подивитися",
            "що подивитись",
            "що там подивитися",
            "що там подивитись",
            "що можна подивитися",
            "що можна побачити",
            "що відвідати",
            "куди сходити",
            "пам'ятк",
            "пам’ятк",
            "визначні місця",
            "цікаві місця"
        ],
        "transport": [
            "як доїхати", "як дістатися",
            "чим доїхати", "транспорт",
            "як добратися"
        ],
        "accommodation": [
            "де зупинитися", "де зупинитись",
            "готел", "хостел",
            "проживан", "ночівл"
        ],
        "budget": [
            "скільки коштує", "скільки грошей",
            "бюджет", "витрат",
            "вартість поїздки"
        ],
        "weather": [
            "коли краще їхати", "коли краще поїхати",
            "який сезон", "погода",
            "пора року"
        ]
    }

    for intent, phrases in keywords.items():
        if any(phrase in low for phrase in phrases):
            return intent

    return None

# Перевірка, чи користувач починає нову подорож
def is_new_recommendation(message, intent, entities):
    if intent != "recommend":
        return False

    # Якщо вказано конкретний напрямок,
    # це може бути уточнення про нього
    if "destination" in entities:
        return False

    low = message.lower()

    new_trip_phrases = [
        "куди поїхати",
        "куди вирушити",
        "де відпочити",
        "порадь напрямок",
        "порадь місце",
        "яке місто обрати",
        "що обрати для відпочинку",
        "хочу кудись поїхати",
        "інше місце",
        "інший напрямок",
        "нову подорож",
        "новий напрямок"
    ]

    return any(
        phrase in low for phrase in new_trip_phrases
    )


# Рекомендації без прив'язки до старого міста
def recommend_destination(memory):
    season = memory.get("season")
    budget = memory.get("budget")
    duration = memory.get("duration")

    if season == "зима":
        candidates = [
            "карпати", "яремче", "львів"
        ]
    elif season == "літо":
        candidates = [
            "карпати", "яремче", "київ"
        ]
    elif duration is not None and duration <= 2:
        candidates = [
            "львів", "яремче",
            "кам’янець-подільський"
        ]
    else:
        candidates = [
            "львів", "київ", "карпати",
            "яремче", "кам’янець-подільський"
        ]

    # Не рекомендуємо попередній напрямок,
    # якщо є інші варіанти
    previous = memory.get("previous_destination")

    alternatives = [
        item for item in candidates
        if item != previous
    ]

    if alternatives:
        candidates = alternatives

    selected = random.choice(candidates)
    info = PLACES[selected]

    # Запам'ятовуємо новий напрямок
    memory["destination"] = selected

    answer = (
        f"Рекомендую {info['title']}. "
        f"Це {info['summary']}."
    )

    if duration:
        answer += (
            f" Можна запланувати поїздку "
            f"на {duration} дн."
        )

    if budget:
        answer += (
            f" Ваш бюджет — {budget} грн. "
            "Перевірте актуальні ціни "
            "на транспорт і проживання."
        )

    return answer


# Створення маршруту
def create_itinerary(info, duration):
    attractions = info.get("attractions", [])

    if not attractions:
        return "Для цього напрямку маршрут ще не підготовлено."

    duration = duration or 2
    duration = max(1, min(duration, 7))

    lines = [
        f"Орієнтовний маршрут: {info['title']}."
    ]

    for day in range(1, duration + 1):
        attraction = attractions[
            (day - 1) % len(attractions)
        ]

        lines.append(
            f"День {day}: {attraction}."
        )

    if duration > len(attractions):
        lines.append(
            "Деякі місця повторюються, "
            "оскільки база пам'яток обмежена."
        )

    lines.append(
        "Це демонстраційний план, "
        "без перевірки графіків роботи."
    )

    return "\n".join(lines)


# Формування відповіді
def respond(intent, memory):
    destination = memory.get("destination")
    info = PLACES.get(destination)

    if intent == "greeting":
        return (
            "Привіт! Я TravelBot.\n"
            "Можу допомогти з вибором напрямку, "
            "пам'ятками, транспортом, бюджетом, "
            "проживанням, кухнею та маршрутом.\n"
            "Напишіть «допомога», щоб побачити можливості."
        )

    if intent == "thanks":
        return "Будь ласка! Гарної подорожі!"

    if intent == "goodbye":
        return "До зустрічі! Приємних мандрівок!"

    if intent == "recommend":
        if info:
            return (
                f"Можна розглянути {info['title']}: "
                f"{info['summary']}.\n"
                "Хочете дізнатися про пам'ятки "
                "або скласти маршрут?"
            )

        return recommend_destination(memory)

    if not info:
        return (
            "Уточніть напрямок, будь ласка: "
            "Львів, Київ, Одеса, Карпати, "
            "Яремче або Кам'янець-Подільський."
        )

    if intent == "attractions":
        attractions = ", ".join(
            info.get("attractions", [])
        )

        return (
            f"У напрямку {info['title']} "
            f"варто відвідати: {attractions}."
        )

    if intent == "transport":
        return (
            f"{info['transport']}\n"
            "Для точного маршруту потрібно знати "
            "місто відправлення. Розклад "
            "перевіряйте у перевізника."
        )

    if intent == "accommodation":
        return (
            f"{info['accommodation']}\n"
            "Ціни та наявність місць "
            "потрібно перевіряти окремо."
        )

    if intent == "weather":
        return (
            f"{info['season']}\n"
            "Це загальна сезонна рекомендація, "
            "а не актуальний прогноз."
        )

    if intent == "budget":
        budget = memory.get("budget")

        answer = (
            f"Для подорожі до напрямку "
            f"«{info['title']}» врахуйте "
            "витрати на транспорт, житло, "
            "харчування та розваги."
        )

        if budget:
            answer += (
                f"\nВаш бюджет: {budget} грн."
            )

        answer += (
            "\nБез дат, кількості туристів "
            "і міста відправлення точну "
            "вартість визначити неможливо."
        )

        return answer

    if intent == "itinerary":
        return create_itinerary(
            info,
            memory.get("duration")
        )

    if intent == "food":
        food = info.get("food", [])

        if not food:
            return (
                "Інформація про місцеву кухню "
                "поки що відсутня."
            )

        return (
            f"У напрямку {info['title']} "
            f"можна скуштувати: {', '.join(food)}."
        )

    if intent == "activities":
        activities = info.get("activities", [])

        if not activities:
            return (
                "Інформація про розваги "
                "поки що відсутня."
            )

        return (
            f"Варіанти дозвілля: "
            f"{', '.join(activities)}."
        )

    return (
        "Не вдалося зрозуміти запит. "
        "Спробуйте переформулювати."
    )


# Довідка
def show_help():
    return (
        "Можливості TravelBot:\n"
        "1. Рекомендація напрямку\n"
        "2. Туристичні пам'ятки\n"
        "3. Бюджет подорожі\n"
        "4. Транспорт\n"
        "5. Проживання\n"
        "6. Сезонні рекомендації\n"
        "7. Маршрут за днями\n"
        "8. Місцева кухня\n"
        "9. Розваги та активності\n\n"
        "Команди:\n"
        "допомога — список можливостей\n"
        "нова подорож — очистити контекст\n"
        "очисти пам'ять — очистити контекст\n"
        "що ти пам'ятаєш — переглянути контекст\n"
        "вихід — завершити роботу"
    )

def detect_context_intent(message, memory):
    low = message.lower().strip()

    # Уточнення працюють, якщо вже відомий напрямок
    if not memory.get("destination"):
        return None

    context_phrases = {
        "attractions": [
            "що там подивитися",
            "що там подивитись",
            "а що там подивитися",
            "а що там подивитись",
            "що там цікавого",
            "а які там пам'ятки",
            "що можна там відвідати"
        ],
        "accommodation": [
            "а де там зупинитися",
            "де там зупинитись",
            "а де переночувати",
            "де там жити"
        ],
        "food": [
            "а що там поїсти",
            "що там скуштувати",
            "яка там кухня"
        ],
        "activities": [
            "а які там розваги",
            "чим там зайнятися",
            "що там робити"
        ],
        "itinerary": [
            "склади маршрут",
            "зроби план поїздки",
            "розпиши по днях"
        ]
    }

    for intent, phrases in context_phrases.items():
        if any(phrase in low for phrase in phrases):
            return intent

    return None

def main():
    memory = {}

    print("TravelBot — туристичний ШІ-консультант")
    print("Для списку команд напишіть «допомога».")

    while True:
        try:
            message = input("\nВи: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTravelBot: До зустрічі!")
            break

        if not message:
            continue

        low = message.lower()

        if low in ["вихід", "exit", "quit"]:
            print("TravelBot: До зустрічі!")
            break

        if low in ["допомога", "help"]:
            print("TravelBot:", show_help())
            continue

        if low in [
            "нова подорож",
            "очисти пам'ять",
            "скинь пам'ять"
        ]:
            memory.clear()
            print(
                "TravelBot: Контекст очищено. "
                "Можемо планувати нову подорож!"
            )
            continue

        if low in [
            "що ти пам'ятаєш",
            "що ти пам’ятаєш",
            "що ти памятаєш",
            "що ти пам'ятаєш?",
            "що ти памятаєш?",
            "покажи пам'ять",
            "покажи пам’ять",
            "покажи память"
        ]:
            print(
                "TravelBot: Поточний контекст:",
                memory if memory else "порожній"
            )
            continue

        # Виділення сутностей
        entities = extract_entities(message)

        # Класифікація наміру
        probabilities = model.predict_proba(
            [message]
        )[0]

        index = probabilities.argmax()
        confidence = float(probabilities[index])

        intent = model.classes_[index]

        # Перевірка ключових слів
        keyword_intent = detect_intent_by_keywords(message)

        # Перевірка уточнень із контекстом
        context_intent = detect_context_intent(message, memory)

        if context_intent is not None:
            intent = context_intent

        elif keyword_intent is not None:
            intent = keyword_intent

        elif confidence < 0.18:
            intent = "unknown"

        # Визначення нового планування
        new_trip = is_new_recommendation(
            message, intent, entities
        )

        if new_trip:
            previous = memory.get("destination")

            # Скидаємо старі параметри подорожі
            memory.clear()

            if previous:
                memory["previous_destination"] = previous

        # Зберігаємо нові сутності
        memory.update(entities)

        print(
            f"[Намір: {intent}; "
            f"впевненість: {confidence:.2f}; "
            f"сутності: {entities}]"
        )

        if intent == "unknown":
            print(
                "TravelBot: Не зовсім зрозумів "
                "повідомлення. Напишіть "
                "«допомога» або уточніть запит."
            )
            continue

        answer = respond(intent, memory)
        print("TravelBot:", answer)

        if intent == "goodbye":
            break


if __name__ == "__main__":
    main()