import pytest
#from unittest.mock import MagicMock, patch
from llm_agent.core_v2 import LLMAgent

# =====================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (Запускают реальную Ollama / API)
# =====================================================================
# Маркируем как 'integration', чтобы их можно было отключать при быстрой проверке

@pytest.mark.integration
def test_calculator_query_live():
    """Реальный запуск агента для проверки математики."""
    # Для тестов лучше использовать локальную модель, если она поднята
    agent = LLMAgent(local=True, ollama_model="qwen3.5:0.8b")
    query = "Сколько будет (5 + 3) * 2? Напиши только цифру."
    
    response = agent.process_query(query)
    
    # Проверяем, что агент смог посчитать и выдать 16
    assert "16" in response


@pytest.mark.integration
def test_football_query_live():
    """Реальный запуск агента для проверки поиска DuckDuckGo."""
    agent = LLMAgent(local=True, ollama_model="qwen3.5:0.8b")
    query = "Кто выиграл последний матч Спартак-Динамо?"
    
    response = agent.process_query(query)
    
    # Проверяем, что в реальном ответе фигурируют названия команд
    assert "Спартак" in response or "Spartak" in response
    assert "Динамо" in response or "Dynamo" in response

# =====================================================================
# ТЕСТЫ ДЛЯ IMAGE_TO_TEXT TOOL (OCR)
# =====================================================================

@pytest.mark.integration
def test_imagetotext_from_url():
    """Тест извлечения текста из изображения по URL."""
    agent = LLMAgent(local=True, ollama_model="qwen3.5:0.8b")
    
    # Используем тестовое изображение с текстом (например, сгенерированное)
    query = "Извлеки текст из изображения: https://example.com/test-image.jpg"
    
    # Проверяем, что агент распознает необходимость использовать инструмент
    plan = agent._ask_llm_for_plan(query)
    assert len(plan) > 0, "План должен содержать хотя бы одно действие"
    assert plan[0]['action'] == 'image_to_text', "Должен использоваться инструмент image_to_text"


def test_imagetotext_tool_direct():
    """Прямой тест инструмента ImageToTextTool с локальным файлом."""
    from llm_agent.tool_imagetotext import ImageToTextTool
    import tempfile
    from PIL import Image, ImageDraw, ImageFont
    
    tool = ImageToTextTool()
    
    # Создаем тестовое изображение с текстом
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        img_path = tmp_file.name
    
    try:
        # Генерируем изображение с текстом
        img = Image.new('RGB', (300, 100), color='white')
        d = ImageDraw.Draw(img)
        # Используем шрифт по умолчанию (простой текст)
        d.text((10, 30), "Привет, мир! Это тест OCR.", fill='black')
        img.save(img_path)
        
        # Извлекаем текст
        result = tool.use(img_path, lang="rus+eng")
        
        # Проверяем, что текст извлечен
        assert "Привет" in result or "тест" in result
        assert "OCR" in result
    finally:
        # Удаляем временный файл
        os.unlink(img_path)


def test_imagetotext_tool_empty_image():
    """Тест обработки пустого/чистого изображения."""
    from llm_agent.tool_imagetotext import ImageToTextTool
    import tempfile
    from PIL import Image
    
    tool = ImageToTextTool()
    
    # Создаем полностью белое изображение
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        img_path = tmp_file.name
    
    try:
        img = Image.new('RGB', (300, 100), color='white')
        img.save(img_path)
        
        result = tool.use(img_path)
        
        # Должен вернуть сообщение, что текст не найден
        assert "текст не найден" in result or "не найден" in result
    finally:
        os.unlink(img_path)


def test_imagetotext_tool_invalid_file():
    """Тест обработки некорректного файла."""
    from llm_agent.tool_imagetotext import ImageToTextTool
    import tempfile
    
    tool = ImageToTextTool()
    
    # Создаем текстовый файл, выдавая его за изображение
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        tmp_file.write(b"This is not an image")
        img_path = tmp_file.name
    
    try:
        result = tool.use(img_path)
        
        # Должен вернуть сообщение об ошибке
        assert "не является корректным изображением" in result or "Ошибка" in result
    finally:
        os.unlink(img_path)


def test_imagetotext_tool_missing_file():
    """Тест обработки отсутствующего файла."""
    from llm_agent.tool_imagetotext import ImageToTextTool
    
    tool = ImageToTextTool()
    result = tool.use("/path/to/non/existent/image.jpg")
    
    assert "не найден" in result or "ошибка" in result.lower()


# Вспомогательная функция для объединения всех тестов
def run_all_image_tests():
    """Запускает все тесты для ImageToTextTool (для удобства)."""
    print("Запуск всех тестов ImageToTextTool...")
    test_imagetotext_from_url()
    test_imagetotext_tool_direct()
    test_imagetotext_tool_empty_image()
    test_imagetotext_tool_invalid_file()
    test_imagetotext_tool_missing_file()
    print("Все тесты пройдены!")
