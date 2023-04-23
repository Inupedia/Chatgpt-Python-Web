import os
import openai
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

# if you have OpenAI API key as an environment variable, enable the below
openai.api_key = os.getenv("OPENAI_API_KEY")


prompt = "你是一个人工智能聊天机器人，你可以与人类进行自然语言对话，理解人类语言中的语义和上下文，并提供有用的信息和回答问题。你可以使用自己的知识库和算法来提供建议、解决问题和回答各种问题，从而帮助人类更好地理解世界。 "


def openai_create(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=[{"role": "system", "content": prompt}]
    )

    return completion.choices[0].message.content


def chatgpt_clone(input, history):
    history = history or []
    s = list(sum(history, ()))
    s.append(input)
    inp = " ".join(s)
    output = openai_create(inp)
    history.append((input, output))
    return history, history


with gr.Blocks() as app:
    gr.Markdown(
        """<h1><center>ChatGPT</center></h1>
    """
    )
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="输入你的问题")
    state = gr.State()
    submit = gr.Button("发送")
    msg.submit(chatgpt_clone, inputs=[msg, state], outputs=[chatbot, state])
    submit.click(chatgpt_clone, inputs=[msg, state], outputs=[chatbot, state])

app.launch(server_port=8080, server_name="0,0,0,0")
