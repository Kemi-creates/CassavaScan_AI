
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model('best_cassavaguard_mobilenet.h5')

CLASS_NAMES = ['CBB', 'CBSD', 'CGM', 'CMD', 'Healthy']

DISEASE_INFO = {
    'CBB': {
        'full_name': 'Cassava Bacteria Blight',
        'symptoms': 'Angular leaf spots, wilting, and stem dieback.',
        'action': 'Remove and destroy infected plants. Avoid working in fields when wet.'
    },
    'CBSD': {
        'full_name': 'Cassava Brown Streak Disease',
        'symptoms': 'Yellow patches on leaves, brown streaks inside the root.',
        'action': 'Use certified clean planting material. Destroy infected crops immediately.'
    },
    'CGM': {
        'full_name': 'Cassava Green Mottle',
        'symptoms': 'Mottled yellowing and distorted leaves.',
        'action': 'Use resistant varieties. Control whitefly populations.'
    },
    'CMD': {
        'full_name': 'Cassava Mosaic Disease',
        'symptoms': 'Mosaic yellowing, distorted and twisted leaves.',
        'action': 'Plant CMD-resistant varieties. Remove and burn infected plants.'
    },
    'Healthy': {
        'full_name': 'Healthy Plant',
        'symptoms': 'No disease detected.',
        'action': 'Your cassava plant appears healthy. Continue normal farming practices.'
    }
}

def predict(image):
    img = Image.fromarray(image).resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array, verbose=0)[0]
    top_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = float(np.max(predictions))
    info = DISEASE_INFO[top_class]
    result = f'''
## { if top_class == 'Healthy' else 'Beware'} {info['full_name']}
**Confidence:** {confidence*100:.1f}%
**Symptoms:** {info['symptoms']}
**Recommended Action:** {info['action']}
---
*CassavaGuard — Built for African smallholder farmers*
    '''
    label_scores = {CLASS_NAMES[i]: float(predictions[i]) for i in range(5)}
    return label_scores, result

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown('''
    # CassavaGuard — Cassava Disease Detector
    ### AI-powered crop disease diagnosis for African farmers
    Upload a photo of a cassava leaf to get an instant diagnosis.
    Built on real field images from African farms.
    ''')
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Cassava Leaf Photo")
            submit_btn = gr.Button("Diagnose", variant="primary")
        with gr.Column():
            label_output = gr.Label(num_top_classes=5, label="Disease Probability")
            result_output = gr.Markdown(label="Diagnosis & Action")
    submit_btn.click(fn=predict, inputs=image_input, outputs=[label_output, result_output])
    gr.Markdown('''
    ### Disease Classes
    | Code | Full Name |
    |---|---|
    | CBB | Cassava Bacteria Blight |
    | CBSD | Cassava Brown Streak Disease |
    | CGM | Cassava Green Mottle |
    | CMD | Cassava Mosaic Disease |
    | — | Healthy |
    ''')

demo.launch()
