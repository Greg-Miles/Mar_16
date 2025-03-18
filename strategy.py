
from settings import API_KEY, MODELS
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
        :param history: История запросов и ответов.
        :param img_path: Путь к изображению для чтения.
        :return: Словарь с ответом от модели ИИ.
        """
        try:
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

        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}", "model": model}
        

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
        :param text: Текст запроса.
        :param model: Модель ИИ, используемая для генерации текста.
        :param history: История запросов и ответов.
        :param image_data: Перекодированное изображение.
        :return: Словарь с ответом от модели ИИ.
        """
        try:      
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

        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}", "model": model}


class ChatFacade:
    """
    Фасад для связи польззователя и моделей ИИ.
    """

    def __init__(self, api_key: str):
        """
        Класс для связи пользователя и моделей ИИ.
        :param api_key: API-ключ для доступа к API.
        :attr text_request: Объект для отправки запроса на генерацию текста.
        :attr image_request: Объект для отправки запроса на чтение изображения.
        :attr history: История запросов и ответов.
        :attr models: Список моделей ИИ.
        :attr strategy: Выбранный режим работы.
        """

        self.text_request = TextRequestStrategy(api_key)
        self.image_request = ImageRequestStrategy(api_key)
        self.history = []
        self.models = MODELS
        self.strategy = ''

    def change_strategy(self, strategy_type: str)-> int:
        """
        Метод для выбора режима работы.
        :param strategy_type: Тип режима работы.
        :raises: ValueError: Если введен некорректный режим работы.
        :return: Выбранный режим работы.
        """

        if strategy_type == 'text' or strategy_type == 'image':
            self.strategy = strategy_type
        else:
            raise ValueError("Некорректный режим работы")

        return self.strategy


    def select_model(self, strategy: str) -> str:
        """
        Метод для выбора модели ИИ.
        :param strategy: Выбранный режим работы.
        :return: Выбранная модель ИИ.
        """
        # Тут можно было бы дать пользователю выбрать модель, но для упрощения задания я выбрал первую модель из списка
        if strategy == 'text':
            model = self.models['text'][0]
        elif strategy == 'image':
            model = self.models['image'][0]
        else:
            raise ValueError("Некорректный режим работы")
        
        return model
    
    def load_image(self, image_path: str) -> str:
        """
        Метод для валидации пути к изображению и его обработки.
        :param image_path: Путь к изображению.
        :raises: FileNotFoundError: Если изображение не найдено.
        :return: Перекодированное изображение.
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
            base64_image = self.load_image(image_path)
            response = self.image_request.execute(text, model, None, base64_image)
        else:
            response = self.text_request.execute(text, model)
        
        self.history.append({text: response})
        return response

    def get_history(self)-> list[dict]:
        """
        Метод для получения истории запросов и ответов.
        :return: Список словарей с запросами и ответами.
        """
        return self.history

    def clear_history(self)-> None:
        """
        Метод для очистки истории запросов и ответов.
        """
        self.history = []

if __name__ == "__main__":
    api_key = API_KEY
    chat = ChatFacade(api_key)
    
    # Смена стратегии
    chat.change_strategy("text")
    
    # Выбор модели
    model = chat.select_model(chat.strategy)
    
    # Отправка текстового запроса
    question = "Расскажите о последних новостях в IT."
    response = chat.ask_question(question, model)
    
    print("Ответ от API:", response)
    
    # Смена стратегии на мультимодальную
    chat.change_strategy("image")
    
    # Выбор модели
    model = chat.select_model(chat.strategy)
    
    # Отправка запроса с изображением
    image_path = "C:\\IT\\python\\homework\\Mar_2\\Mar_2\\Screenshot_13.png"
    question = "Опишите это изображение"

    response = chat.ask_question(question, model, image_path)
    
    print("Ответ от API:", response)
    
    # Просмотр истории запросов
    print("История запросов:", chat.get_history())