import gradio as gr

import asyncio, httpx
import async_timeout

from typing import Optional, List
from pydantic import BaseModel

import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")


class Message(BaseModel):
    role: str
    content: str


async def make_completion(
    messages: List[Message], nb_retries: int = 3, delay: int = 30
) -> Optional[str]:
    """
    Sends a request to the ChatGPT API to retrieve a response based on a list of previous messages.
    """
    header = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    try:
        async with async_timeout.timeout(delay=delay):
            async with httpx.AsyncClient(headers=header) as aio_client:
                counter = 0
                keep_loop = True
                while keep_loop:
                    try:
                        resp = await aio_client.post(
                            url="https://api.openai.com/v1/chat/completions",
                            json={"model": "gpt-3.5-turbo", "messages": messages},
                        )
                        if resp.status_code == 200:
                            return resp.json()["choices"][0]["message"]["content"]
                        else:
                            keep_loop = False
                    except Exception as e:
                        counter = counter + 1
                        keep_loop = counter < nb_retries
    except asyncio.TimeoutError as e:
        logger.error(f"Timeout {delay} seconds !")
    return None


async def predict(input, history):
    """
    Predict the response of the chatbot and complete a running list of chat history.
    """
    if history is None:
        history.append(
            {
                "role": "system",
                "content": "你是一个人工智能聊天机器人，你可以与人类进行自然语言对话，理解人类语言中的语义和上下文，并提供有用的信息和回答问题。你可以使用自己的知识库和算法来提供建议、解决问题和回答各种问题，从而帮助人类更好地理解世界。",
            }
        )
    history.append({"role": "user", "content": input})
    response = await make_completion(history)
    history.append({"role": "assistant", "content": response})
    messages = [
        (history[i]["content"], history[i + 1]["content"])
        for i in range(0, len(history) - 1, 2)
    ]
    return messages, history


"""
Gradio Blocks low-level API that allows to create custom web applications (here our chat app)
"""
with gr.Blocks() as demo:
    gr.Markdown(
        """
        <h1><center>家用ChatGPT</center></h1>
        """
    )
    chatbot = gr.Chatbot(label="ChatGPT")
    state = gr.State([])
    txt = gr.Textbox(show_label=False, placeholder="输入你的问题")
    submit = gr.Button("发送")
    txt.submit(predict, [txt, state], [chatbot, state], lambda x: gr.update(value=""))
    submit.click(predict, [txt, state], [chatbot, state], lambda x: gr.update(value=""))

# demo.launch(server_port=8080, server_name="0,0,0,0")
demo.launch(server_port=8080)
