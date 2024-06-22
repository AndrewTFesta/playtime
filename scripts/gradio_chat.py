"""
@title

@description

"""
import argparse

import gradio as gr
from transformers import pipeline

pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")


def predict(input_img):
    predictions = pipeline(input_img)
    return input_img, {p["label"]: p["score"] for p in predictions}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        label="Select hot dog candidate",
        sources=['upload', 'webcam'],
        type="pil"
    ),
    outputs=[
        gr.Image(label="Processed Image"),
        gr.Label(label="Result", num_top_classes=2)
    ],
    title="Hot Dog? Or Not?",
)

demo.launch(share=True)


def main(main_args):
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    args = parser.parse_args()
    main(vars(args))
