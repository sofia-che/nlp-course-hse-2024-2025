import logging
import json
import numpy as np
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters import Command
from aiogram import F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import asyncio

API_TOKEN = 'my_token'

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

with open('C:/Users/User/Downloads/faq (1).json', 'r', encoding='utf-8') as f:
    data = json.load(f)

kb = [
    [
        KeyboardButton(text="О компании"), 
        KeyboardButton(text="Пожаловаться")
    ]
]

faq_questions = [item['question'] for item in data['faq']]
faq_answers = [item['answer'] for item in data['faq']]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(faq_questions)

sentences = [q.split() for q in faq_questions]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

def sentence_vector(sentence, model):
    words = sentence.split()
    vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

keyboard = ReplyKeyboardMarkup(
    keyboard=kb,  
    resize_keyboard=True,
    input_field_placeholder="Выберите действие"
)

@dp.message(Command("start"))
async def start_command(message: types.Message):
    await message.answer("С чем вам помочь?", reply_markup=keyboard)

@dp.message(F.text == "О компании")
async def about_company(message: types.Message):
    await message.answer("Наша компания занимается доставкой товаров по всей стране.")

@dp.message(F.text == "Пожаловаться")
async def report_issue(message: types.Message):
    await message.reply("Пожалуйста, прикрепите фотографию для жалобы.")

# Обработчик фотографий
@dp.message(F.photo)
async def handle_photo(message: types.Message):
    photo = message.photo[-1]
    file_info = await bot.get_file(photo.file_id)
    file_size = photo.file_size
    file_name = file_info.file_path.split('/')[-1]

    await bot.download(photo.file_id, destination=f"temp_{file_name}")

    await message.reply(f"Название файла: {file_name}\nРазмер файла: {file_size} байт\nВаш запрос передан специалисту.")

@dp.message()
async def handle_message(message: types.Message):
    query = message.text

    if query:
        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix)
        best_match_idx_tfidf = similarities.argmax()
        best_answer_tfidf = faq_answers[best_match_idx_tfidf]

        query_vector = sentence_vector(query, model).reshape(1, -1)
        faq_vectors = np.array([sentence_vector(q, model) for q in faq_questions])
        similarities_w2v = cosine_similarity(query_vector, faq_vectors)
        best_match_idx_w2v = similarities_w2v.argmax()
        best_answer_w2v = faq_answers[best_match_idx_w2v]

        response = (
            f"Метод TF-IDF:\nВопрос: {faq_questions[best_match_idx_tfidf]}\nОтвет: {best_answer_tfidf}\n\n"
            f"Метод Word2Vec:\nВопрос: {faq_questions[best_match_idx_w2v]}\nОтвет: {best_answer_w2v}"
        )

        await message.answer(response)

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())