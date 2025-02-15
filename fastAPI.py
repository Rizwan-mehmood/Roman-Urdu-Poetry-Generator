from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
import numpy as np
import json
import os

app = FastAPI()

MODEL_PATH = "poetry_model/roman_urdu_poetry_model.keras"
CHAR2IDX_PATH = "poetry_model/char2idx.json"
IDX2CHAR_PATH = "poetry_model/idx2char.npy"


data_path = "data/dataset.txt"
with open(data_path, "r", encoding="utf-8") as f:
    text = f.read()
vocab = sorted(set(text))
vocab_size = len(vocab)


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )


model = tf.keras.models.load_model(MODEL_PATH, custom_objects={"loss": loss})
with open(CHAR2IDX_PATH, "r") as f:
    char2idx = json.load(f)
idx2char = np.load(IDX2CHAR_PATH)

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def build_generation_model(trained_model, batch_size=1):
    inputs = tf.keras.layers.Input(batch_shape=(batch_size, None), dtype=tf.int32)
    x = tf.keras.layers.Embedding(vocab_size, 256)(inputs)
    x = tf.keras.layers.LSTM(
        1024,
        return_sequences=True,
        stateful=True,
        recurrent_initializer="glorot_uniform",
    )(x)
    outputs = tf.keras.layers.Dense(vocab_size)(x)

    gen_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    gen_model.set_weights(trained_model.get_weights())
    return gen_model


def generate_text(model, start_string, num_generate=1000, temperature=1.0):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    for layer in model.layers:
        if hasattr(layer, "reset_states"):
            layer.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])

    return start_string + "".join(text_generated)


@app.post("/generate")
def generate_poetry(
    seed_text: str = Form(...), length: int = 50, temperature: float = 0.8
):
    gen_model = build_generation_model(model, batch_size=1)
    generated_poetry = generate_text(
        gen_model, start_string=seed_text, num_generate=500, temperature=0.8
    )
    return {"poetry": generated_poetry}
