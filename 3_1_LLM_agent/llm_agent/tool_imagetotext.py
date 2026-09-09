# llm_agent/tool_imagetotext.py
import os
import tempfile
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import requests
from PIL import Image
import pytesseract


class ImageToTextTool:
    """Инструмент для извлечения текста из изображений с помощью OCR (Tesseract)."""

    name = "image_to_text"
    description = (
        "Извлекает текст из изображений с помощью OCR (Tesseract). "
        "Принимает локальный путь к изображению или URL. Поддерживает форматы: JPG, PNG, BMP, TIFF."
    )

    def use(self, source: str, lang: str = "rus+eng", preprocess: bool = True) -> str:
        """
        Извлекает текст из изображения с помощью OCR.

        Args:
            source: путь к локальному изображению или URL
            lang: языки для распознавания (по умолчанию "rus+eng")
            preprocess: применять предобработку изображения (улучшает качество OCR)

        Returns:
            Строка с извлеченным текстом или сообщение об ошибке.
        """
        try:
            print(f"> Обрабатываю изображение: '{source}'")

            # Определяем, это URL или локальный путь
            is_url = urlparse(source).scheme in ("http", "https")
            temp_file = None

            if is_url:
                print(f"> Скачиваю изображение по URL...")
                response = requests.get(source, timeout=30, stream=True)
                response.raise_for_status()

                # Проверяем Content-Type
                content_type = response.headers.get("Content-Type", "")
                if "image" not in content_type.lower():
                    return f"Ошибка: по URL '{source}' не найдено изображение (Content-Type: {content_type})"

                # Сохраняем во временный файл
                ext = self._get_extension_from_url(source)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
                for chunk in response.iter_content(chunk_size=8192):
                    temp_file.write(chunk)
                temp_file.close()
                image_path = temp_file.name
                print(f"> Изображение сохранено во временный файл: {image_path}")
            else:
                image_path = source
                if not os.path.isfile(image_path):
                    return f"Ошибка: файл '{image_path}' не найден."

            # Проверяем, что файл действительно изображение
            try:
                img = Image.open(image_path)
                print(f"> Размер изображения: {img.size}, формат: {img.format}")
            except Exception as e:
                return f"Ошибка: файл не является корректным изображением. {e}"

            # Предобработка изображения (опционально)
            if preprocess:
                img = self._preprocess_image(img)

            # Извлекаем текст через Tesseract
            print(f"> Запускаю OCR (языки: {lang})...")
            extracted_text = pytesseract.image_to_string(img, lang=lang)

            # Очищаем и форматируем результат
            cleaned_text = extracted_text.strip()
            
            if not cleaned_text:
                return "(текст не найден на изображении)"

            # Формируем результат
            result = (
                f"Извлеченный текст из: {source}\n"
                f"{'=' * 50}\n"
                f"{cleaned_text}\n"
                f"{'=' * 50}\n"
                f"Длина текста: {len(cleaned_text)} символов"
            )

            print(f"> Успешно извлечено {len(cleaned_text)} символов")
            return result

        except pytesseract.TesseractNotFoundError:
            return "Ошибка: Tesseract не установлен. Установите Tesseract OCR (https://github.com/tesseract-ocr/tesseract)"
        except Exception as e:
            print(f"> Ошибка при обработке изображения: {e}")
            return f"Произошла ошибка при извлечении текста из изображения '{source}': {e}"

        finally:
            # Удаляем временный файл, если он был создан
            if temp_file is not None:
                try:
                    os.unlink(temp_file.name)
                except OSError:
                    pass

    def _preprocess_image(self, image):
        """
        Предобработка изображения для улучшения качества OCR.
        """
        from PIL import ImageEnhance, ImageFilter
        
        # Конвертируем в оттенки серого
        if image.mode != 'L':
            image = image.convert('L')
        
        # Увеличиваем контрастность
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Увеличиваем резкость
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Применяем фильтр для улучшения краев
        image = image.filter(ImageFilter.SHARPEN)
        
        return image

    def _get_extension_from_url(self, url: str) -> str:
        """Определяет расширение файла по URL."""
        # Пробуем взять из URL
        path = urlparse(url).path
        ext = os.path.splitext(path)[1]
        if ext and ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
            return ext
        
        # Если не удалось, используем .jpg по умолчанию
        return '.jpg'


# Пример использования:
# tool = ImageToTextTool()
# print(tool.use("https://example.com/image.jpg"))
# print(tool.use("/path/to/local/image.png", lang="eng"))
