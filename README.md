# Information-theoretic stochastic contrastive conditional GAN: InfoSCC-GAN

This repos contains official Pytorch implementation of the paper: **Information-theoretic stochastic contrastive conditional GAN: InfoSCC-GAN**

## Demos

<h2 align="center"><a href="https://info-scc-celeba-10.herokuapp.com/">CelebA demo with 10 attributes</a> | 
<a href="https://info-scc-celeba-15.herokuapp.com/">CelebA demo with 15 attributes</a> |
<a href="https://info-scc-afhq.herokuapp.com/">AFHQ demo</a> </h2>


## Contents

1. [Animal Faces High Quality experiments](#afhq_experiments) <br>
1.1 [Animal Faces Demo](#afhq_demo) <br>
1.2 [EigenGAN experiment](#eigen_gan)  <br>
1.3 [InfoSCC-GAN experiments](#info_scc_gan) <br>
   - [Oneclass global discriminator, Hinge loss, each 2-nd iter classification regularization](#info_scc_gan_global_one_hinge_2) <br>
   - [Oneclass global discriminator, Non saturating loss, each 2-nd iter classification regularization](#info_scc_gan_global_one_non_satur_2)
   - [Multiclass global discriminator, Hinge loss, each 2-nd iteration classification regularization](#info_scc_gan_global_multi_hinge) <br>
   - [Oneclass patch discriminator, Hinge loss, each 2-nd iteration classification regularization](#info_scc_gan_patch_one_hinge) <br>
   - [Oneclass patch discriminator, Non-saturating loss, each 2-nd iteration classification regularization](#info_scc_gan_patch_one_non_saturating) <br>
   - [Oneclass path discriminator, LSGAN loss, each 2-nd iteration classification regularization](#info_scc_gan_patch_one_lsgan)
2. [CelebA experiments](#celeba_experiments) <br>
2.1 [CelebA Demo with 10 attributes](#celeba_10_demo) <br>
2.2 [CelebA Demo with 15 attributes](#celeba_15_demo) <br>
3. [Run experiment](RUN.md)
   
<!-- ################################################################################# 
'Vanilla 'EigenGAN
################################################################################# -->

<h1 align="center"><a name="afhq_experiments"></a> Animal Faces High Quality experiments</h1>

<h2 align="center"><a name="afhq_demo"></a> <a href="https://info-scc-afhq.herokuapp.com/">AFHQ demo</a> </h2>

<h2 align="center"> <a name="eigen_gan"></a> EigenGAN exploration </h2>
<!--
<p align="center"><img src="images/eigen_gan/sample-min.png"></p>
<h4 align="center">Random samples generated using 'vanilla EigenGAN'</h4>

<p align="center"><img src="images/eigen_gan/traverse_L0_D0-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 0, Dimension: 0 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L0_D1-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 0, Dimension: 1 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L0_D2-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 0, Dimension: 2 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L0_D3-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 0, Dimension: 3 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L0_D4-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 0, Dimension: 4 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L0_D5-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 0, Dimension: 5 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L1_D0-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 1, Dimension: 0 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L1_D1-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 1, Dimension: 1 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L1_D2-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 1, Dimension: 2 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L1_D3-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 1, Dimension: 3 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L1_D4-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 1, Dimension: 4 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L1_D5-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 1, Dimension: 5 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L2_D0-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 2, Dimension: 0 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L2_D1-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 2, Dimension: 1 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L2_D2-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 2, Dimension: 2 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L2_D3-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 2, Dimension: 3 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L2_D4-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 2, Dimension: 4 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L2_D5-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 2, Dimension: 5 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L3_D0-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 3, Dimension: 0 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L3_D1-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 3, Dimension: 1 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L3_D2-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 3, Dimension: 2 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L3_D3-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 3, Dimension: 3 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L3_D4-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 3, Dimension: 4 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L3_D5-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 3, Dimension: 5 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L4_D0-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 4, Dimension: 0 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L4_D1-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 4, Dimension: 1 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L4_D2-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 4, Dimension: 2 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L4_D3-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 4, Dimension: 3 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L4_D4-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 4, Dimension: 4 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L4_D5-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 4, Dimension: 5 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L5_D0-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 5, Dimension: 0 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L5_D1-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 5, Dimension: 1 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L5_D2-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 5, Dimension: 2 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L5_D3-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 5, Dimension: 3 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L5_D4-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 5, Dimension: 4 </h4>

<p align="center"><img src="images/eigen_gan/traverse_L5_D5-min.png"></p>
<h4 align="center">EigenGAN <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration. Layer: 5, Dimension: 5 </h4>

-->
<!-- #################################################################################
InfoSCC-GAN. Oneclass discriminator, Hinge loss, classification regularization every 2
################################################################################# -->
<p align="center"><img src="images/eigen_gan/exploration.png"></p>

<h2 align="center"><a name="info_scc_gan"></a>InfoSCC-GAN</h2>

<!-- ##########################################################################################################-->
<h3 align="center"><a name="info_scc_gan_global_one_hinge_2"></a>  Oneclass global discriminator, Hinge loss, each 2-nd iter classification regularization</h3>

<h4 align="center"> <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration </h4>
<p align="center"><img src="images/AFHQ_InfoSCC_global_oneclass_hinge_cls_2/exploration.png"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_global_oneclass_hinge_cls_2/traverse_eps.png"></p>
<p align="center"> Each row: fix input label <img src="https://render.githubusercontent.com/render/math?math=y"> and latent variables <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k">, randomly change <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and  <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_global_oneclass_hinge_cls_2/traverse_eps_zs.png"></p>
<p align="center"> Each row: fix input label <img src="https://render.githubusercontent.com/render/math?math=y">, randomly change latent variables <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> and  <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=y"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_global_oneclass_hinge_cls_2/explore_y.png"></p>
<p align="center"><img src="images/AFHQ_InfoSCC_global_oneclass_hinge_cls_2/explore_y_0.png"></p>
<p align="center"><img src="images/AFHQ_InfoSCC_global_oneclass_hinge_cls_2/explore_y_1.png"></p>
<p align="center"><img src="images/AFHQ_InfoSCC_global_oneclass_hinge_cls_2/explore_y_2.png"></p>
<p align="center">Fix <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k">, explore all <img src="https://render.githubusercontent.com/render/math?math=y"></p>

<!-- ##########################################################################################################-->
<h3 align="center"><a name="info_scc_gan_global_one_non_satur_2"></a> Oneclass global discriminator, Non saturating loss, each 2-nd iter classification regularization</h3>

<h4 align="center"> <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration </h4>
<p align="center"><img src="images/AFHQ_InfoSCC_global_oneclass_non_saturating_cls_2/exploration.png"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_global_oneclass_non_saturating_cls_2/traverse_eps.png"></p>
<p align="center"> Each row: fix input label <img src="https://render.githubusercontent.com/render/math?math=y"> and latent variables <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k">, randomly change <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and  <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_global_oneclass_non_saturating_cls_2/explore_eps_zs.png"></p>
<p align="center"> Each row: fix input label <img src="https://render.githubusercontent.com/render/math?math=y">, randomly change latent variables <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> and  <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=y"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_global_oneclass_non_saturating_cls_2/explore_y_0.png"></p>
<p align="center"><img src="images/AFHQ_InfoSCC_global_oneclass_non_saturating_cls_2/explore_y_1.png"></p>
<p align="center"><img src="images/AFHQ_InfoSCC_global_oneclass_non_saturating_cls_2/explore_y_2.png"></p>
<p align="center">Fix <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k">, explore all <img src="https://render.githubusercontent.com/render/math?math=y"></p>

<!-- ##########################################################################################################-->
<h3 align="center"><a name="info_scc_gan_global_multi_hinge"></a>Multiclass global discriminator, Hinge loss, each 2-nd iter classification regularization</h3>

<h4 align="center"> <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration </h4>
<p align="center"><img src="images/AFHQ_InfoSCC_global_multiclass_hinge_cls_2/exploration.png"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_global_multiclass_hinge_cls_2/traverse_eps.png"></p>
<p align="center"> Each row: fix input label <img src="https://render.githubusercontent.com/render/math?math=y"> and latent variables <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k">, randomly change <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and  <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_global_multiclass_hinge_cls_2/explore_eps_zs.png"></p>
<p align="center"> Each row: fix input label <img src="https://render.githubusercontent.com/render/math?math=y">, randomly change latent variables <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> and  <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=y"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_global_multiclass_hinge_cls_2/explore_y.png"></p>
<p align="center">Fix <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k">, explore all <img src="https://render.githubusercontent.com/render/math?math=y"></p>

<!-- ##########################################################################################################-->
<h3 align="center"><a name="info_scc_gan_patch_one_hinge"></a>One class patch  discriminator, Hinge loss, each 2-nd iter classification regularization</h3>

<h4 align="center"> <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration </h4>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_onelclass_hinge_cls_2/exploration.png"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_onelclass_hinge_cls_2/traverse_eps.png"></p>
<p align="center"> Each row: fix input label <img src="https://render.githubusercontent.com/render/math?math=y"> and latent variables <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k">, randomly change <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and  <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_onelclass_hinge_cls_2/explore_eps_zs.png"></p>
<p align="center"> Each row: fix input label <img src="https://render.githubusercontent.com/render/math?math=y">, randomly change latent variables <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> and  <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=y"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_onelclass_hinge_cls_2/explore_y_0.png"></p>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_onelclass_hinge_cls_2/explore_y_1.png"></p>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_onelclass_hinge_cls_2/explore_y_2.png"></p>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_onelclass_hinge_cls_2/explore_y_3.png"></p>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_onelclass_hinge_cls_2/explore_y_4.png"></p>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_onelclass_hinge_cls_2/explore_y_5.png"></p>
<p align="center">Fix <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k">, explore all <img src="https://render.githubusercontent.com/render/math?math=y"></p>

<!-- ##########################################################################################################-->
<h3 align="center"><a name="info_scc_gan_patch_one_non_saturating"></a>One class patch discriminator, Non-saturating loss, each 2-nd iter classification regularization</h3>

<h4 align="center"> <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration </h4>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_oneclass_non_saturating_cls_2/exploration.png"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_oneclass_non_saturating_cls_2/traverse_eps.png"></p>
<p align="center"> Each row: fix input label <img src="https://render.githubusercontent.com/render/math?math=y"> and latent variables <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k">, randomly change <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and  <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_oneclass_non_saturating_cls_2/explore_eps_zs.png"></p>
<p align="center"> Each row: fix input label <img src="https://render.githubusercontent.com/render/math?math=y">, randomly change latent variables <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> and  <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=y"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_oneclass_non_saturating_cls_2/explore_y_0.png"></p>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_oneclass_non_saturating_cls_2/explore_y_1.png"></p>
<p align="center">Fix <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k">, explore all <img src="https://render.githubusercontent.com/render/math?math=y"></p>

<!-- ##########################################################################################################-->
<h3 align="center"> <a name="info_scc_gan_patch_one_lsgan"></a> One class patch discriminator, LSGAN loss, each 2-nd iter classification regularization</h3>

<h4 align="center"> <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> latent variables exploration </h4>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_oneclass_lsgan_cls_2/exploration.png"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_oneclass_lsgan_cls_2/traverse_eps.png"></p>
<p align="center"> Each row: fix input label <img src="https://render.githubusercontent.com/render/math?math=y"> and latent variables <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k">, randomly change <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and  <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_oneclass_lsgan_cls_2/explore_eps_zs.png"></p>
<p align="center"> Each row: fix input label <img src="https://render.githubusercontent.com/render/math?math=y">, randomly change latent variables <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k"> and  <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"></p>

<h4 align="center"> Explore <img src="https://render.githubusercontent.com/render/math?math=y"></h4>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_oneclass_lsgan_cls_2/explore_y_0.png"></p>
<p align="center"><img src="images/AFHQ_InfoSCC_patch_oneclass_lsgan_cls_2/explore_y_1.png"></p>
<p align="center">Fix <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and <img src="https://render.githubusercontent.com/render/math?math=z_1, ..., z_k">, explore all <img src="https://render.githubusercontent.com/render/math?math=y"></p>

<h1 align="center"><a name="celeba_experiments"></a> CelebA experiments</h1>

<h2 align="center"><a name="celeba_10_demo"></a> <a href="https://info-scc-celeba-10.herokuapp.com/">CelebA demo with 10 attributes</a> </h2>
<h2 align="center"><a name="celeba_15_demo"></a> <a href="https://info-scc-celeba-15.herokuapp.com/">CelebA demo with 15 attributes</a> </h2>
