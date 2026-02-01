import os
from langchain_community.chat_models import ChatOllama
from pypdf import PdfReader
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import LLMChain 

# --- Инициализация модели LLM ---
llm = ChatOllama(model="gemma3:1b", temperature=1)
print("Модель gemma3:1b успешно загружена и готова к работе.")


def load_text_from_file(filepath):
    # Эта функция загружает текст либо из TXT, либо из PDF файла.
    
    if filepath.endswith('.txt'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()

    elif filepath.endswith('.pdf'):
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n" # Заменил \\n на \n для лучшего форматирования
        return text
    
    else:
        # Теперь просто пропускаем неподдерживаемые файлы вместо вызова ошибки
        print(f"Пропускаю файл с неподдерживаемым форматом: {filepath}")
        return None



# Укажите путь к вашей директории с файлами.
# Текущая рабочая директория (там, где запущен скрипт)
DIRECTORY_PATH = "." 

text_vacancy = None
text_resume = None

print(f"Ищу файлы в директории: {os.path.abspath(DIRECTORY_PATH)}")

# Проходим по всем файлам в указанной директории
for filename in os.listdir(DIRECTORY_PATH):
    filepath = os.path.join(DIRECTORY_PATH, filename)
    
    # Проверяем, является ли объект файлом (а не поддиректорией)
    if os.path.isfile(filepath):
        # Загружаем текст из файла
        file_content = load_text_from_file(filepath)
        
        if file_content:
            # Определяем, что это: вакансия или резюме, по имени файла (пример)
            if "vacancy" in filename.lower():
                text_vacancy = file_content
                print(f"Найден файл вакансии: {filename}")
            elif "resume" in filename.lower():
                text_resume = file_content
                print(f"Найден файл резюме: {filename}")

# Проверка, найдены ли оба файла
if not text_vacancy or not text_resume:
    print("Ошибка: Не удалось найти оба файла (вакансии и/или резюме) в указанной директории.")
    exit() # Прерываем выполнение скрипта, если файлы не найдены


# --- Создание Промпта ---
prompt_template = """
Вы — опытный HR-аналитик. Ваша задача — сравнить предоставленное описание вакансии и резюме кандидата. 
Ваш анализ должен быть объективным и основываться только на предоставленном тексте.

**Описание вакансии:**
{vacancy}

**Резюме кандидата:**
{resume}

**Формат вывода:**
Предоставьте результат в следующем формате:
1. **Подходит ли кандидат?**: [Да/Нет/Частично]
2. **Процент соответствия навыков**: [Число от 0% до 100%]
3. **Основные причины (совпадения/несоответствия)**: [Краткий список]
"""
prompt = ChatPromptTemplate.from_template(prompt_template)


# --- Создание Цепочки  ---
chain = LLMChain(llm=llm, prompt=prompt)


# --- Запуск Агента (теперь использует загруженные переменные text_vacancy и text_resume) ---
print("Запускаю процесс анализа локальной моделью gemma3:1b...")

# Вызываем цепочку (chain.invoke), передавая ей фактические данные для плейсхолдеров
response = chain.invoke({"vacancy": text_vacancy, "resume": text_resume})

# Выводим результат работы агента
print("\n--- Результат HR-Агента ---")
print(response['text'])
