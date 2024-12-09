import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
from streamlit_cropper import st_cropper
from PIL import Image
from skimage.data import shepp_logan_phantom
from utils import *

# INISIALISASI #
st.set_page_config(page_title="Simulasi CT Scan", page_icon=':radioactive_sign:', initial_sidebar_state="expanded")

if 'initialized' not in st.session_state:
    st.session_state['initialized'] = False

if 'image' not in st.session_state:
    st.session_state['image'] = None

if st.session_state['initialized'] == False:
    # Tampilan Input
    st.title("Simulasi CT Scan Tranlasi-Rotasi")
    upload1, upload2 = st.columns([0.6, 0.4], gap='large')
    with upload1:
        st.markdown("##### Unggah gambar")
        uploaded_img = st.file_uploader("Unggah gambar", type=['png', 'jpg', 'jpeg'])
        st.divider()
        st.markdown("##### Rancang Phantom")
        st.caption("hanya aktif jika tidak ada gambar terunggah")
        phantom_mode = st.radio('Jenis Phantom', ['Shepp-Logan Phantom', 'Line Phantom'])
        lp_cm = st.slider("Line Pairs/cm", min_value=1.0, max_value=20.0, step=0.5, value = 5.0)
        resolution = st.number_input("Resolution", min_value= 120, max_value = 4320, value=480, step=20)
        
        if uploaded_img == None:
            if phantom_mode == 'Shepp-Logan Phantom':
                image = shepp_logan_phantom()
            else:
                image = create_phantom(resolution, lp_cm, resolution)
        else:
            file_bytes = np.asarray(bytearray(uploaded_img.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            image = process_image(image)                     
    with upload2:
        st.markdown("##### Preview")
        st.image(image, width=1080)
        if st.button("Lanjutkan dengan gambar ini"):
            st.session_state['image'] = image
            st.session_state['initialized'] = True

else:
    # Tampilan Output
    image = st.session_state['image']
    st.title("Simulasi CT Scan Tranlasi-Rotasi")
    image_shape = image.shape

    output1, output2 = st.columns([0.335, 0.665])
    with output1:
        st.markdown("##### Gambar input")
        st.image(image, width=1080)
        st.markdown("##### Sinogram")
        sng_container = st.container()
        # st.pyplot(fig_sng)
    with output2:
        st.markdown("##### Hasil rekonstruksi")
        fbp_container = st.container()
        # st.pyplot(fig_fbp)
    
    with st.sidebar:
        with st.form("Parameters"):
            num_detector = st.number_input('Jumlah Translasi', min_value=10, max_value=image_shape[1], value=100, step=10)
            num_rotation = st.number_input('Jumlah Rotasi', min_value=4, max_value=180, value=60, step=5)
            kVp = st.slider('Kilovolt Peak (kVp) multiplier', min_value=0.8, max_value=1.2, value=1.0, step=0.05)
            mA = st.slider('Miliampere (mA) multiplier', min_value=0.8, max_value=1.2, value=1.0, step=0.05)
            if st.form_submit_button("Proses"):
                _, _,normal_sinogram = generate_sinogram(image, num_detector, num_rotation)
                normal_min = normal_sinogram.min()
                normal_max = normal_sinogram.max()

                thetas, detector_position, sinogram = generate_sinogram(image, num_detector, num_rotation, kVp=kVp, mA=mA)
                sinogram = np.clip(sinogram, normal_min, normal_max)
                Thetas, Detector_Position = np.meshgrid(thetas, detector_position)

                fig_sng, ax_sng = plt.subplots()
                ax_sng.set_xlabel('θ (°)', fontsize=14)
                ax_sng.set_ylabel('r (px)', fontsize=14)
                ax_sng.pcolor(Thetas, Detector_Position, sinogram, shading='auto')
                sng_container.pyplot(fig_sng)
                
                fbp = FBP(sinogram, thetas)
                fbp = cv2.flip(fbp, 0)
                fig_fbp, ax_fbp = plt.subplots()
                ax_fbp.axis('off')
                fig_fbp.gca().set_aspect('equal')
                ax_fbp.pcolor(fbp, cmap='gray')
                fbp_container.pyplot(fig_fbp)
