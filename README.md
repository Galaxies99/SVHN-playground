# SVHN-playground

[[Report: An Exploration of Machine Learning Methods on SVHN Dataset](assets/An%20Exploration%20of%20Machine%20Learning%20Methods%20on%20SVHN%20Dataset.pdf)]

This is a project of CS385: Machine Learning in Shanghai Jiao Tong University.

## Requirements

```bash
pip install -r requirements.txt
```

## Run

For logistic regression based models:

```bash
python run-categorical.py --cfg [Config File]
```

For SVM based models:

```bash
python run-svm.py --cfg [Config File]
```

For deep neural network models:

```bash
python run.py --cfg [Config File]
```

For VAE based generative models:

```bash
python run-vae.py --cfg [Config File]
```

## Other Function

For HOG feature generation (before SVM training):

```bash
python run-feature-gen.py
```

For parameter analysis (after SVM training):

```bash
python run-param-analysis.py --cfg [Config File]
```

For Grad-CAM visualization (after deep neural network training, currently only support ResNet-based networks):

```bash
python run-gradcam.py --cfg [Config File]
```

## Implemented Models

- Logistic regressions;
- Logistic regressions based on HOG features;
- Logistic regressions with Lasso/ridge loss;
- Support vector machines;
- Support vector machines with kernel methods;
- Support vector machines with ridge loss;
- LeNet;
- AlexNet;
- VGG Nets, including vgg11, vgg13, vgg16, vgg19, vgg11bn, vgg13bn, vgg16bn, vgg19bn;
- GoogLeNet;
- ResNets, including resnet18, resnet34, resnet50, resnet101, resnet152;
- Variational auto-encoders;
- beta-VAE;
- Disentangled beta-VAE;
- MSSIM-VAE;
- DFC-VAE.

## Configuration Files

All configuration files of the experiments are in the `configs` folder.

## Citation

```bibtex
@misc{fang2021svhnplayground,
  author =       {Hongjie Fang},
  title =        {An Exploration of Machine Learning Methods on SVHN Dataset},
  howpublished = {\url{https://github.com/Galaxies99/SVHN-playground}},
  year =         {2021}
}
```

## References

1. Netzer, Yuval, et al. "Reading digits in natural images with unsupervised feature learning." (2011).
2. Dalal, Navneet, and Bill Triggs. "Histograms of oriented gradients for human detection." 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR'05). Vol. 1. Ieee, 2005.
3. Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
4. Cortes, Corinna, and Vladimir Vapnik. "Support-vector networks." Machine learning 20.3 (1995): 273-297.
5. Platt, John. "Sequential minimal optimization: A fast algorithm for training support vector machines." (1998).
6. LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.
7. Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012): 1097-1105.
8. Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
9. He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
10. Jacob Gildenblat and contributors, PyTorch library for CAM methods, Available Online: [https://github.com/jacobgil/pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam), GitHub, 2021.
11. Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." Proceedings of the IEEE international conference on computer vision. 2017.
12. Kingma, Diederik P., and Max Welling. "Auto-encoding variational bayes." arXiv preprint arXiv:1312.6114 (2013).
13. Higgins, Irina, et al. "beta-vae: Learning basic visual concepts with a constrained variational framework." (2016).
14. Burgess, Christopher P., et al. "Understanding disentangling in $\beta $-VAE." arXiv preprint arXiv:1804.03599 (2018).
15. Snell, Jake, et al. "Learning to generate images with perceptual similarity metrics." 2017 IEEE International Conference on Image Processing (ICIP). IEEE, 2017.
16. Hou, Xianxu, et al. "Deep feature consistent variational autoencoder." 2017 IEEE Winter Conference on Applications of Computer Vision (WACV). IEEE, 2017.
