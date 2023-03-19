import gradio as gr
import numpy as np
from transformers import pipeline

# 1

# def greet(name, is_morning, temperature):
#     salutation = "Good morning" if is_morning else "Good evening"
#     greeting = f"{salutation} {name}. It is {temperature} degrees today"
#     celsius = (temperature - 32) * 5 / 9
#     return greeting, round(celsius, 2)
#
#
# demo = gr.Interface(
#     fn=greet,
#     inputs=["text", "checkbox", gr.Slider(0, 100)],
#     outputs=["text", "number"],
# )
#
# demo.launch()


# 2

# def sepia(input_img):
#     sepia_filter = np.array([
#         [0.393, 0.769, 0.189],
#         [0.349, 0.686, 0.168],
#         [0.272, 0.534, 0.131]
#     ])
#     sepia_img = input_img.dot(sepia_filter.T)
#     sepia_img /= sepia_img.max()
#     return sepia_img
#
#
# demo = gr.Interface(sepia, gr.Image(shape=(200, 200)), "image")
# demo.launch()


# 3

# def flip_text(x):
#     return x[::-1]
#
#
# def flip_image(x):
#     return np.fliplr(x)
#
#
# with gr.Blocks() as demo:
#     gr.Markdown("Flip text or image files using this demo.")
#     with gr.Tab("Flip Text"):
#         text_input = gr.Textbox()
#         text_output = gr.Textbox()
#         text_button = gr.Button("Flip")
#     with gr.Tab("Flip Image"):
#         with gr.Row():
#             image_input = gr.Image()
#             image_output = gr.Image()
#         image_button = gr.Button("Flip")
#
#     with gr.Accordion("Open for More!"):
#         gr.Markdown("Look at me...")
#
#     text_button.click(flip_text, inputs=text_input, outputs=text_output)
#     image_button.click(flip_image, inputs=image_input, outputs=image_output)
#
# demo.launch()


# 4, 5

# def welcome(name):
#     return f"Welcome to Gradio, {name}!"
#
#
# def increase(num):
#     return num + 1
#
#
# with gr.Blocks() as demo:
#     gr.Markdown(
#     """
#     # Hello World!
#     Start typing below to see the output.
#     """)
#     inp = gr.Textbox(placeholder="What is your name?")
#     out = gr.Textbox()
#     inp.change(welcome, inp, out)
#
#     a = gr.Number(label="a")
#     b = gr.Number(label="b")
#     btoa = gr.Button("a > b")
#     atob = gr.Button("b > a")
#     atob.click(increase, a, b)
#     btoa.click(increase, b, a)
#
# demo.launch()


# 6

asr = pipeline("automatic-speech-recognition", "facebook/wav2vec2-base-960h")
classifier = pipeline("text-classification")


def speech_to_text(speech):
    text = asr(speech)["text"]
    return text


def text_to_sentiment(text):
    return classifier(text)[0]["label"]


# demo = gr.Blocks()
#
# with demo:
#     audio_file = gr.Audio(type="filepath")
#     text = gr.Textbox()
#     label = gr.Label()
#
#     b1 = gr.Button("Recognize Speech")
#     b2 = gr.Button("Classify Sentiment")
#
#     b1.click(speech_to_text, inputs=audio_file, outputs=text)
#     b2.click(text_to_sentiment, inputs=text, outputs=label)


with gr.Blocks() as demo:
    food_box = gr.Number(value=10, label="Food Count")
    status_box = gr.Textbox()
    def eat(food):
        if food > 0:
            return {food_box: food - 1, status_box: "full"}
        else:
            return 0, "hungry"
    gr.Button("EAT").click(
        fn=eat,
        inputs=food_box,
        outputs=[food_box, status_box]
    )


demo.launch()
