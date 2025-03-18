
from settings import API_KEY
import os, base64
#pip install mistralai
from mistralai import Mistral
from abc import ABC, abstractmethod

class RequestStrategy(ABC):

    @abstractmethod
    def execute(self, text: str, model: str, history: list[dict] = None, img_path: str = None) -> dict:
        pass

class TextRequestStrategy(RequestStrategy):
    """
    Класс для отправки запроса на генерацию текста.
    """

    def __init__(self, api_key: str):
        """
        Инициализирует объект TextRequest с указанным API-ключом.
        :param api_key: API-ключ для доступа к API.
        :attr client: Объект клиента Mistral для отправки запросов.
        """

        self.client = Mistral(api_key=api_key)


    def execute(self, text: str, model: str, history: list[dict] = None, img_path: str = None) -> dict:
        """
        Метод для отправки запроса на генерацию текста.
        :param text: Текст запроса.
        :param model: Модель ИИ, используемая для генерации текста.
        :return: Словарь с ответом от модели ИИ.
        """

        chat_response = self.client.chat.complete(
            model=model,
            messages=[{
                "role": "user",
                "content": text
            }]
        )
        return {
            "response": chat_response.choices[0].message.content,
            "model": model
        }

class ImageRequestStrategy(RequestStrategy):
    """
    Класс для отправки запроса на чтение изображения.
    """
    def __init__(self, api_key: str):
        """
        Инициализирует объект ImageRequest с указанным API-ключом.
        :param api_key: API-ключ для доступа к API.
        :attr client: Объект клиента Mistral для отправки запросов.
        """

        self.client = Mistral(api_key=api_key)

    def execute(self, text: str, model: str, history: list[dict] = None, image_data: str = None) -> dict:
        """
        Метод для отправки запроса на составление описания изображения.
        :param image_path: Путь к изображению.
        :param model: Модель ИИ, используемая для генерации текста.
        :return: Словарь с ответом от модели ИИ.
        """


              
        response = self.client.chat.complete(
            model=model,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                        "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    },
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }]
        )
        return {
            "response": response.choices[0].message.content,
            "model": model
        }


class ChatFacade:
    """
    Фасад для связи ползьзователя и моделей ИИ.
    """

    def __init__(self, api_key: str):
        """
        Класс для связи пользователя и моделей ИИ.
        :param api_key: API-ключ для доступа к API.
        :attr text_request: Объект для отправки запроса на генерацию текста.
        :attr image_request: Объект для отправки запроса на чтение изображения.
        :attr history: История запросов и ответов.
        """

        self.text_request = TextRequestStrategy(api_key)
        self.image_request = ImageRequestStrategy(api_key)
        self.history = []
        self.models = ['mistral-large-latest','mistral-small-latest','pixtral-large-latest','pixtral-12b-2409']

    def change_stategy(self, strategy_type: str)-> int:
        """
        Метод для выбора режима работы.
        :raises: ValueError: Если введен некорректный режим работы.
        :return: Выбранный режим работы.
        """

        mode_str = input("""Выберите режим работы: 
                     1 - текст, модель Mistral Large
                     2 - текст, модель Mistral Small
                     3 - изображение, модель Pixtral Large
                     4 - изображение, модель Pixtral 12b
                    """)
        try:
            mode = int(mode_str)
            if mode not in range(1, 5):
                print("Некорректный ввод. Пожалуйста, введите 1 или 2.")
                return self.select_mode()
        except ValueError:
            print("Некорректный ввод. Пожалуйста, введите число.")
            return self.select_mode()
        return mode

    def select_model(self, mode: int) -> str:
        """
        Метод для выбора модели ИИ.
        :param mode: Выбранный режим работы.
        :return: Выбранная модель ИИ.
        """
        
        match mode:
            case 1:
                model = 'mistral-large-latest'
            case 2:
                model = 'mistral-small-latest'
            case 3:
                model = 'pixtral-large-latest'
            case 4:
                model = 'pixtral-12b-2409'
            case _:
                model = 'mistral-large-latest'  # Значение по умолчанию
        
        return model
    
    def load_image(self, image_path: str) -> str:
        """
        Метод для валидации пути к изображению.
        :param image_path: Путь к изображению.
        :raises: FileNotFoundError: Если изображение не найдено.
        :return: Путь к изображению.
        """

        if not os.path.exists(image_path):
            raise FileNotFoundError("Изображение не найдено")
        with open(image_path, 'rb') as image_file:
            # Чтение изображения, таг 'rb' для побитового чтения файла
            image_data = image_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8') # Преобразование изображения в формат base64      
        return base64_image

    def ask_question(self, text: str, model: str, image_path: str = None)-> dict:
        """
        Метод для отправки запроса на генерацию текста или чтение изображения.
        :param text: Текст запроса.
        :param model: Модель ИИ, используемая для генерации текста.
        :param image_path: Путь к изображению.
        :return: Словарь с ответом от модели ИИ.
        """

        if image_path:
            response = self.image_request.execute(image_path, model)
        else:
            response = self.text_request.execute(text, model)
        
        self.history.append({text: response})
        return response

    def get_history(self)-> list[dict]:
        """
        Метод для получения истории запросов и ответов.
        :return: Список кортежей с запросами и ответами.
        """
        return self.history

    def clear_history(self)-> None:
        """
        Метод для очистки истории запросов и ответов.
        """
        self.history = []