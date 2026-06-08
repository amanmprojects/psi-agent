import logging
import os
from turtle import up

import dotenv
from langchain_core.messages import HumanMessage
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent import create_agent

# Init Env variables
dotenv.load_dotenv()

# Creating the agent
agent = create_agent()

# Some logging bs
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat is None:
        return
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="I'm a bot, please talk to me!"
    )


async def messageHandler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat is None or update.message is None:
        return
    chat_id = update.effective_chat.id

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=update.message.text)]},
        config={
            "configurable": {
                "thread_id": str(chat_id),  # for InMemorySaver scoping
                "bot": context.bot,  # telegram Bot object
                "chat_id": chat_id,  # for nodes to send messages
            }
        },
    )

    await context.bot.send_message(chat_id=chat_id, text=result["messages"][-1].content)


if __name__ == "__main__":
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or ""
    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    start_handler = CommandHandler("start", start)

    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), messageHandler)
    application.add_handler(start_handler)
    application.add_handler(message_handler)

    application.run_polling()
