# probe_msg.py —— 自动探测 Msg / TextBlock 字段与 Agent 返回结构
import asyncio
import os

from agentscope.agent import Agent
from agentscope.credential import DeepSeekCredential
from agentscope.message import Msg, TextBlock
from agentscope.model import DeepSeekChatModel
from dotenv import load_dotenv


load_dotenv(override=True)
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY")
MODEL = "deepseek-flash"   # 换成你验证过可用的模型名


def dump_model(cls, title):
    print(f"\n=== {title} ===")
    try:
        for fname, finfo in cls.model_fields.items():
            required = "必填" if finfo.is_required() else f"默认={finfo.default!r}"
            print(f"  {fname:16s} {required}")
    except Exception as e:
        print("  字段探测失败:", e)


async def main():
    dump_model(Msg, "Msg 字段")
    dump_model(TextBlock, "TextBlock 字段")

    agent = Agent(
        name="质检助手",
        system_prompt="你是一名工业质检助手，回答简洁专业。",
        model=DeepSeekChatModel(
            credential=DeepSeekCredential(api_key=DEEPSEEK_API_KEY),
            model=MODEL,
            stream=False,
        ),
    )

    # 按报错提示构造：name 必填，content 传块列表
    user_msg = Msg(
        name="user",
        role="user",
        content=[TextBlock(type="text", text="用一句话介绍你自己")],
    )
    print("\n=== 构造出的 Msg ===")
    print(user_msg)

    print("\n=== 调用 agent.reply() ===")
    reply = await agent.reply(user_msg)
    print("返回类型:", type(reply))
    print("完整对象:", reply)

    for name in ("get_text_content", "get_text", "text"):
        fn = getattr(reply, name, None)
        if callable(fn):
            try:
                print(f"[方法] {name}() ->", fn())
            except Exception as e:
                print(f"[方法异常] {name}() -> {type(e).__name__}: {e}")

    print("[属性] content ->", getattr(reply, "content", None))
    print("[属性] finished_reason ->", getattr(reply, "finished_reason", "<无>"))


asyncio.run(main())