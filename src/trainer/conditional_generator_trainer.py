from typing import Dict, NoReturn
from functools import partial
import copy

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import utils

from .generator_trainer import GeneratorTrainer
from src.models import ResNetSimCLR, LinearClassifier, ConditionalGenerator
from src.models import Discriminator, NLayerDiscriminator
from src.transform import image_generation_augment
from src.utils import accumulate
from src.utils import PathOrStr


class ConditionalGeneratorTrainer(GeneratorTrainer):

    """Trainer for conditional generator"""

    def __init__(self,
                 config_path: PathOrStr,
                 config: Dict):

        super().__init__(config_path, config)

        self._start_step, \
            self._generator, self._discriminator, self._g_ema,\
            self._g_optim, self._d_optim, \
            self._encoder, self._classifier = self._load_model()

        self._d_adv_loss, self._g_adv_loss, self._d_reg_loss, self._cls_loss = self._get_loss()

    def train(self) -> NoReturn:

        """Runs training of the conditional generator"""

        loader = self._get_dl()

        total_steps = self._config['total_steps']
        batch_size = self._config['batch_size']
        cls_reg_every = self._config['cls_reg_every']  # classification consistency regularization
        d_reg_every = self._config['d_reg_every']   # discriminator regularization
        d_reg = self._config['d_reg']   # discriminator regularization parameter
        orth_reg = self._config['orth_reg']  # orthogonal regularization parameter
        log_every = self._config['log_every']
        save_every = self._config['save_every']
        sample_every = self._config['sample_every']
        n_out = self._config['dataset']['n_out']
        ds_name = self._config['dataset']['name']

        log_sample = next(loader)[1] if ds_name == 'celeba' else self._sample_label()
        log_sample = log_sample.to(self._device)

        samples_folder = self._writer.checkpoint_folder.parent / 'samples'
        samples_folder.mkdir(exist_ok=True, parents=True)

        augment = image_generation_augment()
        ema = partial(accumulate, decay=0.5 ** (batch_size / (10 * 1000)))

        # train loop
        for step in (iterator := tqdm(range(total_steps), initial=self._start_step)):

            step = step + self._start_step + 1
            if step > total_steps:
                break

            real_img, real_label = next(loader)
            real_img = real_img.to(self._device)
            real_label = real_label.to(self._device)

            # classification regularization
            if (step - self._start_step - 1) % cls_reg_every == 0:
                self._generator.zero_grad()

                if ds_name not in ['celeba']:
                    real_label_oh = F.one_hot(real_label, num_classes=n_out).float()
                    img_out = self._generator(real_label_oh)
                else:
                    # In case of CelebA, no need to convert to one hot
                    img_out = self._generator(real_label)

                h_out, _ = self._encoder(img_out)
                pred = self._classifier(h_out)

                loss_cls = self._cls_loss(pred, real_label)
                loss_cls.backward()
                self._g_optim.step()

            # D update
            with torch.no_grad():
                # fake_label = self._sample_label()

                fake_label = real_label if ds_name == 'celeba' else self._sample_label()
                fake_img = self._generator(fake_label)

            real_pred = self._discriminator(augment(real_img))
            fake_pred = self._discriminator(augment(fake_img))

            d_loss = self._d_adv_loss(real_pred, fake_pred)
            self._discriminator.zero_grad()
            d_loss.backward()
            self._d_optim.step()

            # D regularize
            if (step - self._start_step - 1) % d_reg_every == 0:
                real_img.requires_grad = True
                real_pred = self._discriminator(augment(real_img))

                r1 = self._d_reg_loss(real_pred, real_img) * d_reg

                self._discriminator.zero_grad()
                r1.backward()
                self._d_optim.step()

            # G update
            fake_label = real_label if ds_name == 'celeba' else self._sample_label()
            # fake_label = self._sample_label()
            fake_img = self._generator(fake_label)
            fake_pred = self._discriminator(augment(fake_img))

            g_loss_adv = self._g_adv_loss(fake_pred)
            g_loss_reg = self._generator.orthogonal_regularizer() * orth_reg
            g_loss = g_loss_adv + g_loss_reg

            self._generator.zero_grad()
            g_loss.backward()
            self._g_optim.step()

            ema(self._g_ema, self._generator)

            # log
            if (step - self._start_step - 1) % log_every == 0:
                self._writer.add_scalar('loss/cls_loss', loss_cls.item(), step)
                self._writer.add_scalar("loss/D", d_loss.item(), step)
                self._writer.add_scalar("loss/D_r1", r1.item(), step)
                self._writer.add_scalar("loss/G", g_loss.item(), step)
                self._writer.add_scalar("loss/G_orth", g_loss_reg.item(), step)
                self._writer.add_scalar("loss/G_adv", g_loss_adv.item(), step)

            if step % sample_every == 0:
                with torch.no_grad():
                    utils.save_image(
                        self._g_ema(log_sample),
                        samples_folder / f'{step:07}.png',
                        nrow=int(batch_size ** 0.5),
                        normalize=True,
                        value_range=(-1, 1),
                    )

            if step % save_every == 0:
                self._save_model(step)

    def _load_model(self):

        lr = eval(self._config['lr'])
        img_size = self._config['dataset']['size']  # size of the images (input and generated)
        n_channels = self._config['dataset']['n_channels']  # number of channels in the images (input and generated)
        n_classes = self._config['dataset']['n_out']  # number of classes
        fine_tune_from = self._config['fine_tune_from']

        # load encoder (pretrained)
        encoder_path = self._config['encoder']['path']
        base_model = self._config['encoder']['base_model']
        out_dim = self._config['encoder']['out_dim']

        encoder = ResNetSimCLR(base_model, n_channels, out_dim)
        ckpt = torch.load(encoder_path)
        encoder.load_state_dict(ckpt)
        encoder = encoder.to(self._device).eval()

        # linear classifier (pretrained)
        classifier_path = self._config['classifier']['path']
        n_feat = self._config['classifier']['n_features']

        classifier = LinearClassifier(n_feat, n_classes)
        ckpt = torch.load(classifier_path)
        classifier.load_state_dict(ckpt)
        classifier = classifier.to(self._device).eval()

        # generator
        z_size = self._config['generator']['z_size']  # size of the input noise
        n_basis = self._config['generator']['n_basis']  # size of the z1, ..., z1 vectors
        noise_dim = self._config['generator']['noise_size']  # size of the noise after adapter, which mixes y and z
        y_type = self._config['generator']['y_type']

        generator = ConditionalGenerator(
            size=img_size,
            y_size=n_classes,
            z_size=z_size,
            out_channels=n_channels,
            n_basis=n_basis,
            noise_dim=noise_dim,
            y_type=y_type
        ).to(self._device).train()
        g_ema = copy.deepcopy(generator).eval()

        # discriminator
        disc_type = self._config['discriminator']['type']

        if disc_type == 'oneclass':
            discriminator = Discriminator(n_channels, img_size)
        elif disc_type == 'patch':
            ndf = self._config['discriminator']['ndf']  # number of filters
            n_layers = self._config['discriminator']['n_layers']
            actnorm = self._config['discriminator']['actnorm']

            discriminator = NLayerDiscriminator(n_channels, ndf, n_layers, use_actnorm=actnorm)
        else:
            raise ValueError('Unsupported discriminator')

        discriminator = discriminator.to(self._device).train()

        # optimizers
        g_optim = optim.Adam(
            generator.parameters(),
            lr=lr,
            betas=(0.5, 0.99),
        )

        d_optim = optim.Adam(
            discriminator.parameters(),
            lr=lr,
            betas=(0.5, 0.99),
        )

        start_step = 0
        if fine_tune_from is not None:
            ckpt = torch.load(fine_tune_from, map_location="cpu")
            start_step = ckpt["step"]

            generator.load_state_dict(ckpt["g"])
            discriminator.load_state_dict(ckpt["d"])
            g_ema.load_state_dict(ckpt["g_ema"])
            g_optim.load_state_dict(ckpt["g_optim"])
            d_optim.load_state_dict(ckpt["d_optim"])
            print(f'Loaded from {fine_tune_from}')

        return start_step, generator, discriminator, g_ema, g_optim, d_optim, encoder, classifier

    def _save_model(self, step: int) -> NoReturn:
        ckpt = {
            'step': step,
            'config': self._config,
            'g': self._generator.state_dict(),
            'd': self._discriminator.state_dict(),
            'g_ema': self._g_ema.state_dict(),
            'g_optim': self._g_optim.state_dict(),
            'd_optim': self._d_optim.state_dict(),
        }

        compute_fid = self._config['fid']

        if compute_fid:
            fid_score = self._compute_fid_score()
            ckpt['fid'] = fid_score
            self._writer.add_scalar('FID', fid_score, step)

            i_score = self._compute_inception_score()
            ckpt['is'] = i_score
            self._writer.add_scalar('IS', i_score, step)

        checkpoint_folder = self._writer.checkpoint_folder
        save_file = checkpoint_folder / f'{step:07}.pt'
        torch.save(ckpt, save_file)
