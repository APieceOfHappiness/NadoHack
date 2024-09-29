from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
import torch
import joblib
import pickle
import numpy as np
from sentence_transformers.util import cos_sim
from dataclasses import dataclass
from razdel import tokenize
from pymystem3 import Mystem


@dataclass
class Feedback:
    class_1: str
    class_2: str
    answer: str


class PredictModel:
    tfidf_high: TfidfVectorizer
    logreg_high: LogisticRegression

    model_level2 = {
        'МОДЕРАЦИЯ': 0,
        'МОНЕТИЗАЦИЯ': 0,
        'УПРАВЛЕНИЕ АККАУНТОМ': 0,
        'ДОСТУП К RUTUBE': None,
        'ПРЕДЛОЖЕНИЯ': 0,
        'ВИДЕО': 0,
        'ТРАНСЛЯЦИЯ': 0,
        'СОТРУДНИЧЕСТВО ПРОДВИЖЕНИЕ РЕКЛАМА': None,
        'ПОИСК': 0,
        'БЛАГОТВОРИТЕЛЬНОСТЬ ДОНАТЫ': None,
    }

    one_class = {
        'ДОСТУП К RUTUBE': 'Приложение\xa0',
        'СОТРУДНИЧЕСТВО ПРОДВИЖЕНИЕ РЕКЛАМА': 'Продвижение канал',
        'БЛАГОТВОРИТЕЛЬНОСТЬ ДОНАТЫ': 'Подключение/отключение донатов',
    }

    model_embed: SentenceTransformer
    llama_model: SentenceTransformer
    faq_embeddings: dict
    faq_answers: dict

    def __init__(self, folder="models/"):
        # Stemming init

        self.m = Mystem()

        # Init high level models

        print("----- INIT START ------")
        self.logreg_high = joblib.load(f'{folder}logreg_high.pkl')
        self.tfidf_high = joblib.load(f'{folder}tfidf_high.pkl')

        print("Level 1 inited")
        # Init second level models

        for key, models in self.model_level2.items():
            if models is not None:
                cur_logreg = joblib.load(f'{folder}{key}_logreg.pkl')
                cur_tfidf = joblib.load(f'{folder}{key}_tfidf.pkl')

                self.model_level2[key] = (cur_tfidf, cur_logreg)

        print("Level 2 inited")
        # Init embedding model

        self.model_embed = SentenceTransformer("ai-forever/sbert_large_nlu_ru")

        with open(f'{folder}embeddings_dict.pkl', 'rb') as f:
            self.faq_embeddings = pickle.load(f)

        with open(f'{folder}answers_dict.pkl', 'rb') as f:
            self.faq_answers = pickle.load(f)

        print("Level 3 inited")
        # Init llama model

        self.DEFAULT_SYSTEM_PROMPT = "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

        self.llama_model = AutoModelForCausalLM.from_pretrained(
            "IlyaGusev/saiga_llama3_8b",
            load_in_8bit=True,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.llama_model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained("IlyaGusev/saiga_llama3_8b")
        self.generation_config = GenerationConfig.from_pretrained("IlyaGusev/saiga_llama3_8b")
        print("Level 4 inited")

    def __text_prep(self, text):
        tokens = [j.text.lower() for j in tokenize(text)]
        return "".join(self.m.lemmatize(' '.join(tokens))).strip()

    def inference(self, sentence):
        sentence_stemming = self.__text_prep(sentence)

        tfidf_high_sent = self.tfidf_high.transform([sentence_stemming])
        pred_level1 = self.logreg_high.predict(tfidf_high_sent)[0]

        # Подсчёт энтропии
        pred_probs = self.logreg_high.predict_proba(tfidf_high_sent)[0]
        entropy = -np.sum(pred_probs * np.log(pred_probs))
        if entropy > 1.25:
            return Feedback(class_1="ОТСУТСТВУЕТ", class_2="Отсутствует", answer="Не знаю")

        tfidf_2lvl_sent = self.model_level2[pred_level1][0].transform([sentence_stemming])
        pred_level2 = self.model_level2[pred_level1][1].predict(tfidf_2lvl_sent)[0]

        question_embedding = self.model_embed.encode(sentence_stemming)

        similarities = cos_sim(question_embedding, self.faq_embeddings[pred_level2])
        answer = self.faq_answers[pred_level2][similarities.argsort()[0, -3:]]
        answer = '\n\n'.join(answer)

        prompt = f"""
        Ты ассистент в сервисе RuTube, который отвечает на вопросы пользователей.
        К тебе пришёл следующий вопрос:
        {sentence}

        Ответь, используя данную информацию:
        {answer}

        Напиши: "Не знаю", если информации недостаточно или она не связана с вопросом 

        Твой краткий ответ:
        """

        prompt = self.tokenizer.apply_chat_template([{
            "role": "system",
            "content": self.DEFAULT_SYSTEM_PROMPT
        }, {
            "role": "user",
            "content": prompt
        }], tokenize=False, add_generation_prompt=True)
        data = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        data = {k: v.to(self.llama_model.device) for k, v in data.items()}
        output_ids = self.llama_model.generate(**data, generation_config=self.generation_config)[0]
        output_ids = output_ids[len(data["input_ids"][0]):]
        output = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        
        return Feedback(class_1=pred_level1, class_2=pred_level2, answer=output)




