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

- Binôme: Kaiji, PAULHIAC, Souleimane, EL QODSI
- Adresses mails: k.paulhiac@universite-paris-saclay.fr, souleimane.el-qodsi@universie-paris-saclay.fr
- [Dépôt GitLab](https://gitlab.dsi.universite-paris-saclay.fr/souleimane.el-qodsi/L1InfoInitiationScienceDonnees-2022-2023-Semaine8)

```{code-cell} ipython3
# Automatically reload code when changes are made
%load_ext autoreload
%autoreload 2
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
%matplotlib inline
import scipy
from scipy import signal
import pandas as pd
import seaborn as sns
from glob import glob as ls
import sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score as sklearn_metric

from utilities import *
from intro_science_donnees import data
from intro_science_donnees import *

images = load_images('data', "*.jpg")
```

+++ {"slideshow": {"slide_type": "slide"}}

## Jeu de données

+++

Notre jeu de données consiste à classifier les images contenant un drapeau croate et les images contenant un drapeau serbe. Nous avons choisi ce jeu de données pour son originalité et sa complexité challengeante. En effet, en comparant un drapeau serbe et un drapeau croate, on remarque que les deux sont très similaires, notamment par les couleurs (bleu, blanc, rouge).

```{code-cell} ipython3
image_grid(images)
```

+++ {"slideshow": {"slide_type": "slide"}}

## Prétraitement

+++

Le prétraitement de nos données a été très difficile, c'est en fait la partie la plus dure de notre projet. En effet, les images que nous avons choisies ne nous ont pas facilité la tâche étant donné qu'elles ne sont pas toutes centrées sur fond uni. La difficulté majeure était donc de trouver un moyen d'isoler le drapeau dans l'image et de rogner l'image une fois ce premier travail effectué. Pour cela, nous avons exploré différentes idées et nous les avons implémentées. Mais ce qui a réellement fonctionné pour nous, c'est d'utiliser plusieurs fonctions auxiliaires pour parvenir à créer un masque binaire de l'image par segmentation des données. En effet, nous avons conçu une fonction drapeau, prenant en paramètre différents seuils et coefficients, et qui se sert d'une fonction appartenance. Cette fonction appartenance détermine, pour un pixel donné de coordonnées x,y, si celui-ci appartient au drapeau ou non, et pour ce faire, elle utilise les données des pixels voisins grâce à une fonction blockCreation, qui crée un bloc de pixels centré autour d'un pixel entré en paramètres. Cependant, la fonction appartenance ne travaille que sur les composantes rouge et bleu, et par conséquent, il a fallu créer une autre fonction de binarisation d'image par seuillage :  une fonction whiteness transformant l'image en un tableau NumPy de booléens valant vrai si le pixel est blanc, faux sinon. Une autre fonction très similaire à celle-ci a permis de perfectionner le résultat de drapeau() : une fonction yellowness, faisant le même travail que whiteness à savoir évaluer si chaque pixel est jaune ou non, car en effet, les drapeaux croate et serbe contiennent des petites zones jaunes, surtout le drapeau serbe au niveau de l'écusson. Finalement, pour rogner les images, nous nous sommes servis du masque binaire que la fonction drapeau a créé pour nous, pour chercher la zone la plus concentrée en pixels valant vrai, autrement dit, la zone la plus susceptible d'être le drapeau en lui-même, et nous avons utilisé la méthode crop de la bibliothèque PIL, à laquelle on a passé en paramètre ce que l'on a estimé être les frontières de cette zone de fortes concentration de booléens valant vrai.

+++ {"slideshow": {"slide_type": "slide"}}

## Visualisation des données

+++

L'étape de visualisation nous a permis d'avoir une vue d'ensemble des  données pour mieux comprendre leur structure. Dans notre projet, nous avons appliqué cette étape en utilisant des graphiques et des matrices de corrélation pour explorer les différentes caractéristiques de notre jeu de données. Nous avons utilisé des classificateurs tels que le KNN pour regrouper les images en fonction de leurs caractéristiques et avons ensuite visualisé les résultats de cette classification.

Comme mentionné précédemment, les matrices de corrélation, sur la table standardisée, nous a permis de constater que certains attributs étaient fortement corrélées, tandis que d'autres étaient faiblement corrélées, comme par exemple redness et blueness (forte corrélation négative) ou class et blueness (faible corrélation : le bleu ne permet pas de distinguer les deux classes) .
L'étape de visualisation nous a aussi permis de déterminer le nombre optimal d'attributs, en observant un graphique de la performance du modèle en fonction du nombre d'attributs, et nous avons observé que le nombre d'attributs n'affecte pas fortement les performances, néanmoins, on peut souligner qu'un nombre d'attributs entre 1 et 3 ou 19 et 21 garantit des performances optimales.

+++ {"slideshow": {"slide_type": "slide"}}

## Classificateurs favoris

+++

Les algorithmes de classification favoris que nous retenons sont l'AdaBoost et le NeuralNet car leurs performances sur l'ensemble d'entraînement et l'ensemble de test dépassent celles des autres classificateurs.

+++ {"slideshow": {"slide_type": "slide"}}

## Résultats

### Observations

+++

On a pu observer déjà que dans le jeu de données sans attributs/brut, les résultats sont similaires aux résultats sur la table prétraitée (validation croisée) (avg training/test avec données brutes : 0.71/0.51 -> avg training/test avec données prétraitées 0.74/0.47).
Sur notre ensemble d'attributs larges, on obtient de même des résultats similaires (0.78/0.52). En revanche, avec 6 attributs ad-hoc, les résultats sont beaucoup plus performants (0.89/0.84) et similaires à lorsque l'on fait une ANOVA (0.87/0.81). Ainsi, globalement, le modèle est beaucoup plus performant avec 6 attributs ad-hoc ou avec 5 attributs par analyse de variance univariée. En utilisant des classificateurs, on observe que les résultats sont presque parfaits pour les algorithmes de classification AdaBoost et NeuralNet (0.96/0.96 et 1/0.94)

+++

### Interprétations

+++

Les attributs ad-hoc permettent de radicalement booster la classification des deux drapeaux, les classificateurs sont très peformants. On peut en déduire que notre prétraitement a été bien effectué et les attributs bien choisis.

+++ {"slideshow": {"slide_type": "slide"}, "jp-MarkdownHeadingCollapsed": true}

## Discussion 

Comme mentionné précédemment, la difficulté majeure était de prétraiter des images quasi-aléatoirement choisies sur Internet, de taille/luminosité/arrière-plan/résolution différent-e(s), ce qui nous a forcé à fournir le plus gros effort du projet sur cette partie-là, mais le résultat final était assez satisafaisant, nos fonctions drapeau et crop_flag sont lentes puisque la quantité de calcul est conséquente mais leur performance ne déçoit pas sur la plupart des images. <br> <br>
En outre, nous avons conscience que notre projet pourrait être utilisé dans la vraie vie, notamment à des buts de surveillance. En effet, il peut être utilisé pour suivre l'utilisation des drapeaux sur les médias sociaux ou sur les sites Web. Cette information peut être utilisée pour surveiller les sentiments nationalistes, les tendances politiques ou les activités transfrontalières potentiellement sensibles. Le passé politique de la Serbie et de la Croatie rend l'utilisation de notre projet particulièrement pertinente. La région des Balkans a été le théâtre de conflits  violents au cours des dernières décennies. Les tensions entre les Serbes et les Croates ont souvent été liées à l'utilisation de symboles nationaux tels que les drapeaux. Pendant la guerre en Croatie de 1991 à 1995, les forces serbes ont utilisé le drapeau serbe comme symbole de leur campagne militaire en Croatie. Le drapeau serbe est donc souvent considéré comme un symbole de l'agression serbe en Croatie. D'un autre côté, le drapeau croate est souvent associé à l'indépendance de la Croatie vis-à-vis de la Yougoslavie et est utilisé comme symbole de la nation croate. Ainsi, la classification d'images des drapeaux serbes et croates peut être utilisée pour surveiller les tensions entre les deux pays et pour identifier les mouvements nationalistes. Elle peut également aider les autorités à prévenir les conflits potentiels en surveillant l'utilisation des drapeaux dans les manifestations et les rassemblements politiques.

+++

## Conclusion

+++

Le projet présenté a nécessité un prétraitement de données difficile, en particulier pour isoler le drapeau dans les images sélectionnées. Le masque binaire obtenu par segmentation des données a permis de rogner les images pour obtenir la zone la plus susceptible d'être le drapeau. Les matrices de corrélation et la visualisation ont permis de déterminer le nombre optimal d'attributs pour une performance optimale. Les algorithmes de classification les plus performants sont l'AdaBoost et le NeuralNet. Les résultats ont montré que le modèle est beaucoup plus performant avec 6 attributs ad-hoc ou avec 5 attributs par analyse de variance univariée. Les classificateurs AdaBoost et NeuralNet donnent des résultats presque parfaits pour les deux drapeaux.
