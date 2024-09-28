import requests
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters
from configs import FLASK_SERVER_URL, TELEGRAM_TOKEN
from database.requests import add_to_db


async def handle_message(update: Update, context):
    user_message = update.message.text
    responsed = ''
    await update.message.reply_text("Загрузка...")

    try:
        response = requests.post(FLASK_SERVER_URL, json={"question": user_message})
        if response.status_code == 200:
            responsed = response.json()
            server_answer = responsed.get('answer', 'Извините, произошла ошибка.')
            if server_answer == "Не знаю":
                message = "Извините, я не могу ответить на ваш вопрос"
            else:
                message = f"{responsed['class1']}\r\n{responsed['class2']}\r\n\r\n{server_answer}"
            await update.message.reply_text(message)
        else:
            await update.message.reply_text(f"Ошибка: {response.status_code}")
    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка: {str(e)}")

    add_to_db(question=user_message, class1=responsed.get('class1'), class2=responsed.get('class2'), answer=responsed.get('answer'))


async def start_command(update: Update, context):
    await update.message.reply_text("Привет! Я ваш бот поддержки на Rutube. Я здесь, чтобы помочь вам разобраться с любыми вопросами, связанными с использованием платформы. Задавайте вопросы — будь то проблемы с видео, настройки аккаунта или просто советы по работе с Rutube. Я постараюсь помочь вам быстро и эффективно!")


def main():
    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    application.run_polling()


if __name__ == '__main__':
    main()
