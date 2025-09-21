import os, json, numpy as np
from PIL import Image, ImageOps
import streamlit as st

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 

import tensorflow as tf
import keras
from keras.applications.resnet50 import preprocess_input

tf.config.optimizer.set_jit(False)
tf.config.optimizer.set_experimental_options({"disable_meta_optimizer": True})

CLASS_NAMES = ["Dark", "Green", "Light", "Medium"]
IMG_SIZE = 224

for meta_name in ["coffee_meta.json", "resnet50_coffee_final.meta.json"]:
    if os.path.exists(meta_name):
        try:
            meta = json.load(open(meta_name))
            CLASS_NAMES = meta.get("class_names", CLASS_NAMES)
            IMG_SIZE = int(meta.get("img_size", IMG_SIZE))
            break
        except Exception:
            pass

st.set_page_config(page_title="Coffee Bean Classifier", page_icon="☕")
st.title("☕ Coffee Bean Classifier")
st.caption(f"Uploads resized to {IMG_SIZE}×{IMG_SIZE} · Classes: {', '.join(CLASS_NAMES)}")

@st.cache_resource
def load_model_and_flags():
    m = keras.models.load_model(
        "resnet50_coffee_final.keras",
        compile=False,
        custom_objects={
            "preprocess_input": preprocess_input,   
            "resnet50_preproc": preprocess_input,   
        },
        safe_mode=False,
    )

    names = [l.name.lower() for l in m.layers]
    has_preproc = any(
        ("preproc" in n) or ("rescaling" in n) or ("normalization" in n) for n in names
    )

    _ = m.predict(np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32"), verbose=0)
    return m, has_preproc

model, HAS_PREPROC = load_model_and_flags()
st.caption(f"Model internal preprocessing: {'ON' if HAS_PREPROC else 'OFF'}")

def prepare(img: Image.Image):
    img = ImageOps.exif_transpose(img).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.asarray(img).astype("float32")
    if not HAS_PREPROC:
        x = preprocess_input(x)
    return x[None, ...] 

file = st.file_uploader("Upload an image", type=["jpg","jpeg","png","webp"])
if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded", use_column_width=True)

    x = prepare(img)
    raw = model.predict(x, verbose=0)[0]
    probs = tf.nn.softmax(raw).numpy()
    top = int(np.argmax(probs))

    st.subheader(f"Prediction: **{CLASS_NAMES[top]}**")
    st.write(f"Confidence: {probs[top]:.2%}")
    st.bar_chart({c: float(p) for c, p in zip(CLASS_NAMES, probs)})
