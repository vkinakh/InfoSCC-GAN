# import streamlit as st
# from streamlit_drawable_canvas import st_canvas
# import os
# from PIL import Image
#
# st.set_page_config("VAE Playground")
# st.title("🎮 VAE Playground")
# # title_img = Image.open("images/title_img.png")
#
# # st.image(title_img)
# st.markdown(
#     "This is a simple streamlit app to showcase how different VAEs "
#     "function and how the differences in architecture and "
#     "hyperparameters will show up in the generated images. \n \n"
#     "In this playground there will be two scenarios that you can use to "
#     "interact with the models:"
# )
# st.markdown(
#     """
#     1. **Image Reconstruction:** <br>
#     Observe quality of image reconstruction
#     2. **Image Interpolation:** <br>
#     Sample images equally spaced between 2 drawn images. Observe tradeoff
#     between image reconstruction and latents space regularity
#     """, unsafe_allow_html=True
# )
#
# #  the first is a reconstruction one which is "
# # "used to look at the quality of image reconstruction. The second one is "
# # "interpolation where you can generate intermediary data points between "
# # "two images. From here this you can analyze the regularity of the latent "
# # "distribution."
#
# st.markdown(
#     "There are also two different architectures. The first one is the vanilla "
#     "VAE and the other is the convolutional VAE which uses convolutional layers"
#     " for the encoder and decoder. "
#     "To find out more check this accompanying"
#     " [blogpost](https://towardsdatascience.com/beginner-guide-to-variational-autoencoders-vae-with-pytorch-lightning-13dbc559ba4b)"
# )
# st.subheader("Hyperparameters:")
# st.markdown(
#     "- **alpha**: Weight for reconstruction loss, higher values will lead to better"
#     "image reconstruction but possibly poorer generation \n"
#     "- **dim**: Hidden Dimension of the model."
# )
#
#
# def load_model_files():
#     files = os.listdir("./configs/")
#     # Docker creates some whiteout files which mig
#     files = [i for i in files if ".yml" in i]
#     clean_names = files
#     return {k: v for k, v in zip(clean_names, files)}
#
#
# file_name_map = load_model_files()
# files = list(file_name_map.keys())
#
# st.header("🖼️ Image Reconstruction", "recon")
#
# with st.form("reconstruction"):
#     model_name = st.selectbox("Choose Model:", files,
#                               key="recon_model_select")
#     recon_model_name = file_name_map[model_name]
#     recon_canvas = st_canvas(
#         # Fixed fill color with some opacity
#         fill_color="rgba(255, 165, 0, 0.3)",
#         stroke_width=8,
#         stroke_color="#FFFFFF",
#         background_color="#000000",
#         update_streamlit=True,
#         height=150,
#         width=150,
#         drawing_mode="freedraw",
#         key="recon_canvas",
#     )
#     submit = st.form_submit_button("Perform Reconstruction")
#     if submit:
#         # recon_model = utils.load_model(recon_model_name)
#         # inp_tens = utils.canvas_to_tensor(recon_canvas)
#         # _, _, out = recon_model(inp_tens)
#         # out = (out+1)/2
#         # out_img = utils.resize_img(utils.tensor_to_img(out), 150, 150)
#         print('Keke')
# if submit:
#     # st.image(out_img)
#     print('Lele')
#
#
# st.header("🔍 Image Interpolation", "interpolate")
# with st.form("interpolation"):
#     model_name = st.selectbox("Choose Model:", files)
#     inter_model_name = file_name_map[model_name]
#     stroke_width = 8
#     cols = st.beta_columns([1, 3, 2, 3, 1])
#
#     with cols[1]:
#         canvas_result_1 = st_canvas(
#             # Fixed fill color with some opacity
#             fill_color="rgba(255, 165, 0, 0.3)",
#             stroke_width=stroke_width,
#             stroke_color="#FFFFFF",
#             background_color="#000000",
#             update_streamlit=True,
#             height=150,
#             width=150,
#             drawing_mode="freedraw",
#             key="canvas1",
#         )
#
#     with cols[3]:
#         canvas_result_2 = st_canvas(
#             # Fixed fill color with some opacity
#             fill_color="rgba(255, 165, 0, 0.3)",
#             stroke_width=stroke_width,
#             stroke_color="#FFFFFF",
#             background_color="#000000",
#             update_streamlit=True,
#             height=150,
#             width=150,
#             drawing_mode="freedraw",
#             key="canvas2",
#         )
#     submit = st.form_submit_button("Perform Interpolation")
#     if submit:
#         # inter_model = utils.load_model(inter_model_name)
#         # inter_tens1 = utils.canvas_to_tensor(canvas_result_1)
#         # inter_tens2 = utils.canvas_to_tensor(canvas_result_2)
#         #inter_output = utils.perform_interpolation(
#         #    inter_model, inter_tens1, inter_tens2
#         # )
#         print('22222')
# if submit:
#     #st.image(inter_output)
#     print('111111')
#
# st.write(
#     """
#     ## 💡 Interesting Note:
#     At low values of alpha, we can see the phenomenon known as the **posterior
#     collapse**. This is when the loss function does not weight reconstruction
#     quality sufficiently and the reconstructed images look like digits but
#     nothing like the input.
#
#     Essentially what happens is that the encoder encodes data points to a
#     random gaussian distribution (to minimize KL Loss) but this does not give
#     sufficient information to the decoder. In this case our decoder behaves
#     very similarly to a Generative Adversarial Network (GAN) which generates
#     images from random noise.
#     """
# )

import streamlit as st

import numpy as np

import torch

from src.models import ConditionalGenerator

device = 'cuda'


@st.cache(allow_output_mutation=True)
def load_model():
    print('Loading model')
    model_path = './runs/Sep06_21-23-15_gallager_celeba_generation_sel_columns_10/checkpoint/0200000.pt'

    size = 128
    y_size = n
    z_size = noise_dim = 512
    out_channels = 3
    n_basis = 6

    g_ema = ConditionalGenerator(size, y_size, z_size, out_channels, n_basis, noise_dim)
    ckpt = torch.load(model_path)
    g_ema.load_state_dict(ckpt['g_ema'])
    g_ema.eval().to(device)
    return g_ema


@st.cache(allow_output_mutation=True)
def get_eps_zs(model: ConditionalGenerator):
    eps = model.sample_eps(1)
    zs = model.sample_zs(1)
    return eps, zs

st.title('InfoSCC-GAN CelebA demo')

# Define input y
columns = ['Bald', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair',
            'Mustache', 'Wearing_Hat', 'Eyeglasses','Wearing_Necktie', 'Double_Chin']

n = len(columns)
input_y = [0] * n

for i in range(n):
    input_y[i] = int(st.checkbox(columns[i]))

model = load_model()
eps, zs = get_eps_zs(model)

input_tensor = torch.FloatTensor(input_y).unsqueeze(0).to(device)

with torch.no_grad():
    img = model(input_tensor, eps, zs).squeeze().cpu()

img = (img.permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8)
st.image(img, caption='Generated image', width=400)
