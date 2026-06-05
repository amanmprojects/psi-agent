import logging

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent import agent

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!"
    )

async def messageHandler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    initial_state =


if __name__ == "__main__":
    application = (
        ApplicationBuilder()
        .token("8965863543:AAG0ZjkBihF07RBaIp5i4HYBWsAvA97pf54")
        .build()
    )

    start_handler = CommandHandler("start", start)

    message_handler = MessageHandler(filters.TEXT, messageHandler)
    application.add_handler(start_handler)
    application.add_handler(message_handler)

    application.run_polling()
