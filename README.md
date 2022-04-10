# GraphGAN

The general
idea is that we have list of features (orange box) that are relevant
for capturing dependence inside a halo (dashed red box) and the
tidal fields that are relevant for capturing the dependence outside
of halos (dashed purple). Then, these inputs are fed into the GANGenerator(crimson box) where it tries to learn the desired output
labels (yellow box). At the end the output from the GAN-Generator,
together with the input, are fed into the GAN-Discriminator (blue
box) to determine the performance of the GAN-Generator.

![alt text](https://github.com/melon-lemon/GraphGAN/blob/main/diag_2d_f.png?raw=true)

## Dependencies

To run the example notebook, the following Python packages are required other than standard ones like pandas, numpy, scipy, matplotlib:

* [astropy](http://www.astropy.org)
* [TensorFlow](https://www.tensorflow.org/)



## Contact and reporting issues
For questions/issues regarding this repo please use the Issues feature by clicking on "New Issues"
