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

## Citation

```bibtex
@misc{fang2021svhnplayground,
  author =       {Hongjie Fang},
  title =        {An Exploration of Machine Learning Methods on SVHN Dataset},
  howpublished = {\url{https://github.com/Galaxies99/SVHN-playground}},
  year =         {2021}
}
```