# GraphGAN

The general
idea is that we have list of features (orange box) that are relevant
for capturing dependence inside a halo (dashed red box) and the
tidal fields that are relevant for capturing the dependence outside
of halos (dashed purple). Then, these inputs are fed into the GANGenerator(crimson box) where it tries to learn the desired output
labels (yellow box). At the end the output from the GAN-Generator,
together with the input, are fed into the GAN-Discriminator (blue
box) to determine the performance of the GAN-Generator.

