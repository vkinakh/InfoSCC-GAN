import streamlit as st

import numpy as np

import torch

from src.models import ConditionalGenerator

device = 'cuda'


@st.cache(allow_output_mutation=True)
def load_model():
    print('Loading model')
    model_path = './runs/Sep04_17-23-58_wyner_celeba_generation_cls_4/checkpoint/0200000.pt'

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
columns = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips',
           'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
           'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
           'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
           'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
           'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

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
