# Installation
## Conda installation
```bash
conda env create -f environment.yml
```

# Training

<!-- ## Training of the encoder
TODO -->

## Training of the classifier
To run the training of the classifier, first fill the **config file**. Examples of detailed config files for Animal 
Faces High Quality (AFHQ) and CelebA available are available: `configs/afhq_classification.yml` and `configs/celeba_classification.yml`.

Then run
```bash
python main.py --mode train --task classification --config <path to config file>
```

## Training of the generator
To run the training of the generator, first fill the **config file**. Examples of detailed config files for Animal Faces
High Quality (AFHQ) and CelebA available are available: `configs/afhq_generation.yml` and `configs/celeba_generation.yml`.

Then run
```bash
python main.py --mode train --task generation --config <path to config file>
```

# Evaluation

<!-- ## Evaluation of the encoder
TODO -->

<!-- ## Evaluation of the classifier
TODO -->

## Evaluation of the generator

Evaluation of the generator includes: compute FID score, compute Inception Score (IS), compute Chamfer distance, 
perform attribute control accuracy, traverse z1, ... zk variables, explore epsilon variable.

To run the evaluation, first fill the **config file**, put path to the generator in **fine_tune_from** field.
Then run
```bash
python main.py --mode evaluate --task generation --config <path to config file>
```

# Run the demo

Demos are built using Streamlit.

To run Animal Faces High Quality (AFHQ) demo run:
```bash
streamlit run afhq_demo.py
```

To run CelebA with 10 attributes (`Bald`, `Black_Hair`, `Blond_Hair`, `Brown_Hair`, `Gray_Hair`, `Mustache`, 
`Wearing_Hat`, `Eyeglasses`, `Wearing_Necktie`, `Double_Chin`)

```bash
streamlit run celeba_demo_10_labels.py
```

To run CelebA with 15 attributes (`Bald`, `Blurry`, `Chubby`, `Double_Chin`, `Eyeglasses`, `Goatee`, `Gray_Hair`, 
`Mustache`, `Narrow_Eyes`, `Pale_Skin`, `Receding_Hairline`, `Rosy_Cheeks`, `Sideburns`, `Wearing_Hat`, 
`Wearing_Necktie`)

```bash
streamlit run celeba_demo_15_labels.py
```

# Pretrained models

|Dataset|Model type|Download link|
|-------|----------|-------------|
|AFHQ   |Encoder   |[Download](https://drive.google.com/file/d/1Go0CgmiNLoTIm3y0R_bzpzNMLzynJmqd/view?usp=sharing)|
|AFHQ   |Classifier (3 classes)|[Download](https://drive.google.com/file/d/1IxC2Rke-JNoprNmVwC135dweDvQQK324/view?usp=sharing)|
|AFHQ   |Generator (3 classes)|[Download](https://drive.google.com/file/d/1vRNEVS65xWx6_m9sFnbtP1k-083uy4bQ/view?usp=sharing)|
|CelebA |Encoder   |[Download](https://drive.google.com/file/d/1cMFAKkUwLG3imumHcgNo-b4zC8JOCEj4/view?usp=sharing)|
|CelebA |Classifier (10 attributes)|[Download](https://drive.google.com/file/d/1PjKCkfFTfeUSXi6rgqbsa-4V5oobiSLg/view?usp=sharing)|
|CelebA |Classifier (15 attributes)|[Download](https://drive.google.com/file/d/1koWqPXbxQxlkgYKyjtsNAQnlRqETNe1c/view?usp=sharing)|
|CelebA |Generator (10 attributes) |[Download](https://drive.google.com/file/d/1mctcnfvocoLDV0sz657FOUEtRsbhQ8WC/view?usp=sharing)|
|CelebA |Generator (15 attributes) |[Download](https://drive.google.com/file/d/1VQZrLQI9M_Lm16HAFL5bzqGHz1EOnnNU/view?usp=sharing)|
