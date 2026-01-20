from fastai.vision.all import *
import gradio as gr
from PIL import Image

model = load_learner('models/road_sign_classifier_resnet_r34_best.pkl')

road_sign_labels = model.dls.vocab

def recognize_image(image):
    img= image.resize((192, 192))
    pred, idx, probs = model.predict(img)
    return dict(zip(road_sign_labels, map(float, probs)))

image = gr.Image(type="pil",
                height=400,    
                width=700)
label = gr.Label()
examples = [
    'test_images/unknown_01.png',
    'test_images/unknown_02.jpg',
    'test_images/unknown_03.jpg',
    'test_images/unknown_04.jpg',
    'test_images/unknown_05.jpg',
    'test_images/unknown_06.jpg'
    ]
iface = gr.Interface(fn=recognize_image, inputs=image, outputs=label, examples=examples)
iface.launch(inline=False,debug=True)