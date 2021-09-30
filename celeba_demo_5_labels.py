import streamlit as st
import numpy as np
import torch
from src.models import ConditionalGenerator

device = 'cuda'


@st.cache(allow_output_mutation=True)
def load_model():
    print('Loading model')
    model_path = './runs/Sep12_16-03-53_wyner_celeba_generation_sel_columns_2/checkpoint/0200000.pt'

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
columns = ['Bald', 'Mustache', 'Wearing_Hat', 'Eyeglasses', 'Wearing_Necktie']

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
