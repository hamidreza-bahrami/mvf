import streamlit as st
import numpy as np
import cv2
import time
from keras.models import load_model
from keras.preprocessing.image import img_to_array

model = load_model('model.h5')

def preprocess_image(image):
    file_bytes = np.array(bytearray(image.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = cv2.resize(img, (128, 128))
    img = img.astype('float32') / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def classify_image(img):
    y_pred = model.predict(img)[0][0]

    if y_pred > 0.5:
        gender = "مرد"
        confidence = y_pred * 100
    else:
        gender = "زن"
        confidence = (1 - y_pred) * 100

    return gender, confidence

def show_page():
    st.write("<h1 style='text-align: center; color: blue;'>مدل تشخیص چهره مرد و زن</h1>", unsafe_allow_html=True)
    st.write("<h2 style='text-align: center; color: gray;'>Convolutional Neural Network Model</h2>", unsafe_allow_html=True)
    st.write("<h4 style='text-align: center; color: gray;'>Robo-Ai.ir طراحی شده توسط</h4>", unsafe_allow_html=True)
    st.link_button("Robo-Ai بازگشت به", "https://robo-ai.ir")
    
    image = st.file_uploader("**تصویر موردنظر را بارگذاری کنید:**", type=['jpg', 'jpeg', 'png'])   

    if image is not None:
        st.image(image, caption="**تصویر ورودی**", use_column_width=True)
        
        if st.button("🔍 تحلیل تصویر"):
            with st.chat_message("assistant"):
                with st.spinner("🔄 در حال تحلیل تصویر..."):
                    time.sleep(3)  # Simulate processing time
                    
                    img = preprocess_image(image)
                    predicted_label, confidence = classify_image(img)

                    st.success("✅ تحلیل انجام شد")
                    result_text = f"🔹 بر اساس ارزیابی من، تصویر **{predicted_label}** است.\n📊 **درصد اطمینان:** {confidence:.2f}%"
                    
                    def stream_text(text):
                        for word in text.split(" "):
                            yield word + " "
                            time.sleep(0.09)
                    
                    st.write_stream(stream_text(result_text))

show_page()
