import math
import numpy as np
import streamlit as st
import torch
import torch.nn.functional as F

from src.models import ConditionalGenerator


# Global settings
device = 'cpu'
size = 256
n_channels = 3
classes = ['Cat', 'Dog', 'Wild']
n_classes = len(classes)
z_size = noise_dim = 512
n_layers = int(math.log2(size) - 2)
n_basis = 6
y_size = n_classes
y_type = 'one_hot'
bs = 16  # number of samples generate
n_cols = int(math.sqrt(bs))
model_path = './models/AFHQ/afhq_generator.pt'


@st.cache(allow_output_mutation=True)
def load_model(model_path: str) -> ConditionalGenerator:

    print(f'Loading model: {model_path}')
    g_ema = ConditionalGenerator(size, y_size, z_size, n_channels, n_basis, noise_dim)
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    g_ema.load_state_dict(ckpt['g_ema'])
    g_ema.eval().to(device)
    return g_ema


@st.cache(allow_output_mutation=True)
def get_eps(model: ConditionalGenerator, n: int) -> torch.Tensor:
    eps = model.sample_eps(n)
    return eps.to(device)


st.title('InfoSCC-GAN AFHQ demo')
st.markdown('This demo shows *Stochastic Contrastive Conditional Generative Adversarial Network* (InfoSCC-GAN) '
            'in action. You can conditionally generate samples from the specific class and change the generated '
            'images using latent space exploration')

st.subheader(r'<- Use sidebar to explore $z_1, ..., z_k$ latent variables')
label_str = st.radio('Select animal type', options=classes)

model = load_model(model_path)
eps = get_eps(model, bs)

# get label
label = torch.tensor(classes.index(label_str))
label_one_hot = F.one_hot(label, num_classes=n_classes).float()

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
label_input = label_one_hot.unsqueeze(0).repeat(bs, 1).to(device)

with torch.no_grad():
    imgs = model(label_input, eps, zs_torch).squeeze(0).cpu()
imgs = [(imgs[i].permute(1, 2, 0).numpy() * 127.5 + 127.5).astype(np.uint8) for i in range(bs)]


counter = 0
for r in range(bs // n_cols):
    cols = st.columns(n_cols)

    for c in range(n_cols):
        cols[c].image(imgs[counter])
        counter += 1
