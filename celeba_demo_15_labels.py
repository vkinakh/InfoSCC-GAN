import math
import streamlit as st
import numpy as np
import torch

from src.models import ConditionalGenerator
from src.utils import get_device

device = get_device()
size = 128
n_channels = 3
columns = ['Bald', 'Blurry', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair', 'Mustache', 'Narrow_Eyes',
           'Pale_Skin', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns', 'Wearing_Hat', 'Wearing_Necktie']
n = len(columns)
y_size = n
z_size = noise_dim = 512
n_layers = int(math.log2(size) - 1)
n_basis = 6
bs = 16
n_cols = 4
y_type = 'multi_label'
path = './models/CelebA/celeba_generator_15.pt'


@st.cache(allow_output_mutation=True)
def load_model(model_path: str) -> ConditionalGenerator:
    print('Loading model')
    g_ema = ConditionalGenerator(size, y_size, z_size, n_channels, n_basis, noise_dim)
    ckpt = torch.load(model_path)
    g_ema.load_state_dict(ckpt['g_ema'])
    g_ema.eval().to(device)
    return g_ema


@st.cache(allow_output_mutation=True)
def get_eps(model: ConditionalGenerator, n_img: int) -> torch.Tensor:
    eps = model.sample_eps(n_img)
    return eps.to(device)


st.title('InfoSCC-GAN CelebA demo')
st.markdown('This demo shown *stochastic Contrastive Conditional Generative Adversarial Network* (InfoSCC-GAN) '
            'in action. You can conditionally generate samples with the specific attributes and change the generated'
            'images using latent space exploration. Attributes are in range [0, 1], select them using slider')

st.subheader(r'<- Use sidebar to explore $z_1, ..., z_k$ latent variables')
input_y = [0] * n

# Slider
for i in range(n):
    input_y[i] = st.slider(columns[i], max_value=1., min_value=0., step=0.1)

model = load_model(path)
eps = get_eps(model, bs)

# label
input_label = torch.FloatTensor(input_y).unsqueeze(0).to(device)

# get zs
zs = np.array([[0.0] * n_basis] * n_layers, dtype=np.float32)

for l in range(n_layers):
    st.sidebar.markdown(f'## Layer: {l}')
    for d in range(n_basis):
        zs[l][d] = st.sidebar.slider(f'Dimension: {d}', key=f'{l}{d}',
                                     min_value=-5., max_value=5., value=0., step=0.1)

st.markdown(r'Click on __Change eps__ button to change input $\varepsilon$ latent space')
change_eps = st.button('Change eps')
if change_eps:
    eps = model.sample_eps(bs).to(device)

zs_torch = torch.from_numpy(zs).unsqueeze(0).repeat(bs, 1, 1).to(device)
input_label = input_label.repeat(bs, 1).to(device)

with torch.no_grad():
    imgs = model(input_label, eps, zs_torch).squeeze(0).cpu()
imgs = [(imgs[i].permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8) for i in range(bs)]


counter = 0
for r in range(bs // n_cols):
    cols = st.columns(n_cols)

    for c in range(n_cols):
        cols[c].image(imgs[counter])
        counter += 1
