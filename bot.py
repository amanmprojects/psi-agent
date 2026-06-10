import logging
import os

import dotenv
from langchain_core.messages import HumanMessage
from telegram import Update
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent import create_agent

# Init Env variables
dotenv.load_dotenv()

# Some logging bs
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)


async def post_init(application: Application) -> None:
    agent, conn = await create_agent()
    application.bot_data["agent"] = agent
    application.bot_data["db_conn"] = conn


async def post_shutdown(application: Application) -> None:
    conn = application.bot_data.get("db_conn")
    if conn:
        await conn.close()
        print("Database connection closed and WAL checkpointed.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat is None:
        return
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text="Hi👋\nWhat can I do for you?"
    )


async def messageHandler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_chat is None or update.message is None:
        return
    chat_id = update.effective_chat.id

    agent = context.application.bot_data["agent"]

    await agent.ainvoke(
        {"messages": [HumanMessage(content=update.message.text)]},
        config={
            "configurable": {
                "thread_id": str(chat_id),
                "bot": context.bot,
                "chat_id": chat_id,
            }
        },
    )


if __name__ == "__main__":
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or ""
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)
        .post_shutdown(post_shutdown)
        .build()
    )

    start_handler = CommandHandler("start", start)
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), messageHandler)
    application.add_handler(start_handler)
    application.add_handler(message_handler)

    application.run_polling()
