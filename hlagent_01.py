import asyncio
import os

from agentscope.agent import Agent
from agentscope.console import launch_console
from agentscope.credential import DeepSeekCredential
from agentscope.model import DeepSeekChatModel
from dotenv import load_dotenv


load_dotenv(override=True)
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")

async def main() -> None:
    agent = Agent(
        name="质检助手",
        system_prompt="你是一名工业质检助手，回答简洁专业。",
        model=DeepSeekChatModel(
            credential=DeepSeekCredential(
                api_key=DEEPSEEK_API_KEY,
            ),
            model="deepseek-flash",
        ),
    )
    await launch_console(agent)


asyncio.run(main())