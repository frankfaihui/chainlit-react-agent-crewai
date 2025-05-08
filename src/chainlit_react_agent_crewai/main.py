import chainlit as cl

@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(content="Hello, how can I help you today?").send()
