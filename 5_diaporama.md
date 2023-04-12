---
jupytext:
  notebook_metadata_filter: rise
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
rise:
  auto_select: first
  autolaunch: false
  backimage: fond.png
  centered: false
  controls: false
  enable_chalkboard: true
  height: 100%
  margin: 0
  maxScale: 1
  minScale: 1
  scroll: true
  slideNumber: true
  start_slideshow_at: selected
  transition: none
  width: 90%
---

+++ {"slideshow": {"slide_type": "slide"}}

# Serbie et croatie

- Binôme: Kaihi, PAULHIAC, Souleimane, EL QODSI
- Adresses mails: k.paulhiac@universite-paris-saclay.fr, souleimane.el-qodsi@universie-paris-saclay.fr
- [Dépôt GitLab](https://gitlab.dsi.universite-paris-saclay.fr/souleimane.el-qodsi/L1InfoInitiationScienceDonnees-2022-2023-Semaine8)

+++ {"slideshow": {"slide_type": "slide"}}

## Jeu de données

+++

Notre jeu de données consiste à classifier les images contenant un drapeau croate et les images contenant un drapeau serbe. Nous avons choisi ce jeu de données pour son originalité et sa complexité challengeante. En effet, en comparant un drapeau serbe et un drapeau croate, on remarque que les deux sont très similaires, notamment par les couleurs (bleu, blanc, rouge).

```{code-cell} ipython3
from utilities import *
from intro_science_donnees import data
from intro_science_donnees import *

images = load_images('data', "*.jpg")
image_grid(images)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Prétraitement

+++ {"slideshow": {"slide_type": "slide"}}

## Visualisation des données

+++ {"slideshow": {"slide_type": "slide"}}

## Classificateurs favoris

+++ {"slideshow": {"slide_type": "slide"}}

## Résultats

### Observations

+++

### Interprétations

+++ {"slideshow": {"slide_type": "slide"}}

## Discussion 

Vous pourriez parler *par exemple* des biais potentiels de vos données, de l'utilisation d'un tel projet dans la vraie vie, des difficultés rencontrées et de comment vous les avez surmontées (ou pas).

+++

## Conclusion
