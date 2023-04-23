import gradio as gr
import openai

import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


async def make_completion(history):
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=history)
    return completion.choices[0].message.content


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
        <h1><center>家用聊天机器人</center></h1>
        """
    )
    chatbot = gr.Chatbot(label="ChatGPT")
    state = gr.State([])
    txt = gr.Textbox(show_label=False, placeholder="输入你的问题")
    submit = gr.Button("发送")
    txt.submit(predict, [txt, state], [chatbot, state])
    submit.click(predict, [txt, state], [chatbot, state])

demo.launch(server_port=8080, server_name="0.0.0.0")
# demo.launch(server_port=8080)
