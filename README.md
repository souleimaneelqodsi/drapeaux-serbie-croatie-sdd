# Serbie et Croatie / Serbia and Croatia

- Binôme / Team: Kaiji, PAULHIAC, Souleimane, EL QODSI
- Adresses mails / Email addresses: k.paulhiac@universite-paris-saclay.fr, souleimane.el-qodsi@universie-paris-saclay.fr
- [Dépôt GitLab / GitLab Repository](https://gitlab.dsi.universite-paris-saclay.fr/souleimane.el-qodsi/L1InfoInitiationScienceDonnees-2022-2023-Semaine8)

## Mentions importantes / Important Mentions

### Français
Ce projet a été réalisé **en binôme** dans le cadre du projet d'introduction à la science des données du 2ème semestre de licence portail mathématiques-informatique à l'université Paris-Saclay.
Les concepteurs et auteurs de la structure initiale du projet, des fichiers Markdown/Jupyter Notebook, de nombreuses cellules de code et de texte, ainsi que de nombreuses fonctions dans utilities.py et du fichier index.md sont : [Nicolas Thiéry](https://nicolas.thiery.name) et Fanny Pouyet. Ces éléments sont soumis à licence CC-BY-SA. Les modifications et ajouts ultérieurs ont été réalisés par Kaiji PAULHIAC et Souleimane EL QODSI. 
Le projet est lisible sur Jupyter(Lab) uniquement.

### English
This project was carried out **as a team** as part of the data science introduction project in the second semester of the Mathematics-Computer Science bachelor's program at Paris-Saclay University.
The initial structure of the project, Markdown/Jupyter Notebook files, many code and text cells, as well as many functions in utilities.py and the index.md file were designed and authored by [Nicolas Thiéry](https://nicolas.thiery.name) and Fanny Pouyet. These elements are licensed under CC-BY-SA. Subsequent modifications and additions were made by Kaiji PAULHIAC and Souleimane EL QODSI.
The project is readable on Jupyter(Lab) only.

## Jeu de données / Dataset

### Français
Notre projet consiste à classifier les images contenant un drapeau croate et les images contenant un drapeau serbe. Nous avons choisi ce jeu de données pour son originalité et sa complexité challengeante. En effet, en comparant un drapeau serbe et un drapeau croate, on remarque que les deux sont très similaires, notamment par les couleurs (bleu, blanc, rouge).

### English
Our project involves classifying images containing a Croatian flag and images containing a Serbian flag. We chose this dataset for its originality and challenging complexity. Indeed, comparing a Serbian flag and a Croatian flag, we notice that they are very similar, especially in colors (blue, white, red).

## Prétraitement / Preprocessing

### Français
Le prétraitement de nos données a été très difficile, c'est en fait la partie la plus dure de notre projet. En effet, les images que nous avons choisies ne nous ont pas facilité la tâche étant donné qu'elles ne sont pas toutes centrées sur fond uni. La difficulté majeure était donc de trouver un moyen d'isoler le drapeau dans l'image et de rogner l'image une fois ce premier travail effectué. Pour cela, nous avons exploré différentes idées et nous les avons implémentées. Mais ce qui a réellement fonctionné pour nous, c'est d'utiliser plusieurs fonctions auxiliaires pour parvenir à créer un masque binaire de l'image par segmentation des données. En effet, nous avons conçu une fonction drapeau, prenant en paramètre différents seuils et coefficients, et qui se sert d'une fonction appartenance. Cette fonction appartenance détermine, pour un pixel donné de coordonnées x,y, si celui-ci appartient au drapeau ou non, et pour ce faire, elle utilise les données des pixels voisins grâce à une fonction blockCreation, qui crée un bloc de pixels centré autour d'un pixel entré en paramètres. Cependant, la fonction appartenance ne travaille que sur les composantes rouge et bleu, et par conséquent, il a fallu créer une autre fonction de binarisation d'image par seuillage :  une fonction whiteness transformant l'image en un tableau NumPy de booléens valant vrai si le pixel est blanc, faux sinon. Une autre fonction très similaire à celle-ci a permis de perfectionner le résultat de drapeau() : une fonction yellowness, faisant le même travail que whiteness à savoir évaluer si chaque pixel est jaune ou non, car en effet, les drapeaux croate et serbe contiennent des petites zones jaunes, surtout le drapeau serbe au niveau de l'écusson. Finalement, pour rogner les images, nous nous sommes servis du masque binaire que la fonction drapeau a créé pour nous, pour chercher la zone la plus concentrée en pixels valant vrai, autrement dit, la zone la plus susceptible d'être le drapeau en lui-même, et nous avons utilisé la méthode crop de la bibliothèque PIL, à laquelle on a passé en paramètre ce que l'on a estimé être les frontières de cette zone de fortes concentration de booléens valant vrai.

### English
The preprocessing of our data was very difficult; it was actually the hardest part of our project. The images we chose were not centered on a uniform background, making it challenging to isolate the flag and crop the image. To achieve this, we explored and implemented various ideas. What really worked for us was using multiple auxiliary functions to create a binary mask of the image through data segmentation. We designed a function called `drapeau`, which takes various thresholds and coefficients as parameters, and uses a function called `appartenance`. This `appartenance` function determines, for a given pixel with coordinates x,y, whether it belongs to the flag or not, using neighboring pixel data through a `blockCreation` function, which creates a block of pixels centered around an input pixel. However, `appartenance` only works on the red and blue components, so we had to create another thresholding function: `whiteness`, which transforms the image into a NumPy array of booleans, true if the pixel is white, false otherwise. Another similar function, `yellowness`, evaluates whether each pixel is yellow or not, as Croatian and Serbian flags contain small yellow areas, especially the Serbian flag in the crest area. Finally, to crop the images, we used the binary mask created by the `drapeau` function to find the area most concentrated in true values, likely the flag itself, and used the `crop` method from the PIL library, passing the estimated borders of this high concentration area as parameters.

## Visualisation des données / Data Visualization

### Français
L'étape de visualisation nous a permis d'avoir une vue d'ensemble des  données pour mieux comprendre leur structure. Dans notre projet, nous avons appliqué cette étape en utilisant des graphiques et des matrices de corrélation pour explorer les différentes caractéristiques de notre jeu de données. Nous avons utilisé des classificateurs tels que le KNN pour regrouper les images en fonction de leurs caractéristiques et avons ensuite visualisé les résultats de cette classification.

### English
The visualization step allowed us to get an overview of the data to better understand its structure. In our project, we applied this step using charts and correlation matrices to explore the different characteristics of our dataset. We used classifiers such as KNN to group images based on their features and then visualized the results of this classification.

### Français
Comme mentionné précédemment, les matrices de corrélation, sur la table standardisée, nous a permis de constater que certains attributs étaient fortement corrélées, tandis que d'autres étaient faiblement corrélées, comme par exemple redness et blueness (forte corrélation négative) ou class et blueness (faible corrélation : le bleu ne permet pas de distinguer les deux classes) .
L'étape de visualisation nous a aussi permis de déterminer le nombre optimal d'attributs, en observant un graphique de la performance du modèle en fonction du nombre d'attributs, et nous avons observé que le nombre d'attributs n'affecte pas fortement les performances, néanmoins, on peut souligner qu'un nombre d'attributs entre 1 et 3 ou 19 et 21 garantit des performances optimales.

### English
As mentioned earlier, correlation matrices on the standardized table allowed us to see that some attributes were strongly correlated, while others were weakly correlated, such as redness and blueness (strong negative correlation) or class and blueness (weak correlation: blue does not distinguish between the two classes).
The visualization step also allowed us to determine the optimal number of attributes by observing a graph of model performance as a function of the number of attributes. We noted that the number of attributes does not strongly affect performance; however, we can highlight that having between 1 and 3 or 19 and 21 attributes guarantees optimal performance.

## Classificateurs favoris / Favorite Classifiers

### Français
Les algorithmes de classification favoris que nous retenons sont l'AdaBoost et le NeuralNet car leurs performances sur l'ensemble d'entraînement et l'ensemble de test dépassent celles des autres classificateurs.

### English
The favorite classification algorithms we retain are AdaBoost and NeuralNet because their performance on the training set and the test set exceeds that of other classifiers.

## Résultats / Results

### Observations

### Français
On a pu observer déjà que dans le jeu de données sans attributs/brut, les résultats sont similaires aux résultats sur la table prétraitée (validation croisée) (avg training/test avec données brutes : 
0.71/0.51 -> avg training/test avec données prétraitées 0.74/0.47).
Sur notre ensemble d'attributs larges, on obtient de même des résultats similaires (0.78/0.52). En revanche, avec 6 attributs ad-hoc, les résultats sont beaucoup plus performants (0.89/0.84) et similaires à lorsque l'on fait une ANOVA (0.87/0.81). Ainsi, globalement, le modèle est beaucoup plus performant avec 6 attributs ad-hoc ou avec 5 attributs par analyse de variance univariée. En utilisant des classificateurs, on observe que les résultats sont presque parfaits pour les algorithmes de classification AdaBoost et NeuralNet (0.96/0.96 et 1/0.94)

### English
We observed that in the dataset without attributes/raw data, the results are similar to those on the preprocessed table (cross-validation) (avg training/test with raw data: 0.71/0.51 -> avg training/test with preprocessed data: 0.74/0.47).
On our large attribute set, we get similarly comparable results (0.78/0.52). However, with 6 ad-hoc attributes, the results are much better (0.89/0.84) and similar to when we perform an ANOVA (0.87/0.81). Overall, the model is much more performant with 6 ad-hoc attributes or with 5 attributes from univariate variance analysis. Using classifiers, we see that the results are nearly perfect for the AdaBoost and NeuralNet classification algorithms (0.96/0.96 and 1/0.94).

### Interprétations / Interpretations

### Français
Les attributs ad-hoc permettent de radicalement booster la classification des deux drapeaux, les classificateurs sont très peformants. On peut en déduire que notre prétraitement a été bien effectué et les attributs bien choisis.

### English
The ad-hoc attributes significantly boost the classification of the two flags, and the classifiers are very performant. We can deduce that our preprocessing was well done and the attributes well chosen.

## Discussion 

### Français
Comme mentionné précédemment, la difficulté majeure était de prétraiter des images quasi-aléatoirement choisies sur Internet, de taille/luminosité/arrière-plan/résolution différent-e(s), ce qui nous a forcé à fournir le plus gros effort du projet sur cette partie-là, mais le résultat final était assez satisfaisant, mais malgré tout nous avons quand même eu à modifier des images manuellement. Nos fonctions drapeau et crop_flag sont lentes puisque la quantité de calcul est conséquente mais leur performance ne déçoit pas sur la plupart des images. <br> <br>
En outre, nous avons conscience que notre projet pourrait être utilisé dans la vraie vie, notamment à des buts de surveillance. En effet, il peut être utilisé pour suivre l'utilisation des drapeaux sur les médias sociaux ou sur les sites Web. Cette information peut être utilisée pour surveiller les sentiments nationalistes, les tendances politiques ou les activités transfrontalières potentiellement sensibles. Le passé politique de la Serbie et de la Croatie rend l'utilisation de notre projet particulièrement pertinente. La région des Balkans a été le théâtre de conflits  violents au cours des dernières décennies. Les tensions entre les Serbes et les Croates ont souvent été liées à l'utilisation de symboles nationaux tels que les drapeaux. Pendant la guerre en Croatie de 1991 à 1995, les forces serbes ont utilisé le drapeau serbe comme symbole de leur campagne militaire en Croatie. Le drapeau serbe est donc souvent considéré comme un symbole de l'agression serbe en Croatie. D'un autre côté, le drapeau croate est souvent associé à l'indépendance de la Croatie vis-à-vis de la Yougoslavie et est utilisé comme symbole de la nation croate. Ainsi, la classification d'images des drapeaux serbes et croates peut être utilisée pour surveiller les tensions entre les deux pays et pour identifier les mouvements nationalistes. Elle peut également aider les autorités à prévenir les conflits potentiels en surveillant l'utilisation des drapeaux dans les manifestations et les rassemblements politiques.

### English
As mentioned earlier, the major difficulty was preprocessing images almost randomly chosen from the Internet, with different sizes/brightness/backgrounds/resolutions, which forced us to put most of our effort into this part of the project. The final result was quite satisfactory, but we still had to manually modify some images. Our `drapeau` and `crop_flag` functions are slow because the amount of computation is substantial, but their performance does not disappoint on most images. <br> <br>
Moreover, we are aware that our project could be used in real life, particularly for surveillance purposes. Indeed, it can be used to monitor the use of flags on social media or websites. This information can be used to track nationalist sentiments, political trends, or potentially sensitive cross-border activities. The political history of Serbia and Croatia makes our project's use particularly relevant. The Balkan region has been the scene of violent conflicts in recent decades. Tensions between Serbs and Croats have often been linked to the use of national symbols such as flags. During the war in Croatia from 1991 to 1995, Serbian forces used the Serbian flag as a symbol of their military campaign in Croatia. The Serbian flag is therefore often considered a symbol of Serbian aggression in Croatia. On the other hand, the Croatian flag is often associated with Croatia's independence from Yugoslavia and is used as a symbol of the Croatian nation. Thus, the classification of Serbian and Croatian flag images can be used to monitor tensions between the two countries and identify nationalist movements. It can also help authorities prevent potential conflicts by monitoring flag usage in protests and political gatherings.

## Conclusion

### Français
Le projet présenté a nécessité un prétraitement de données difficile, en particulier pour isoler le drapeau dans les images sélectionnées. Le masque binaire obtenu par segmentation des données a permis de rogner les images pour obtenir la zone la plus susceptible d'être le drapeau. Les matrices de corrélation et la visualisation ont permis de déterminer le nombre optimal d'attributs pour une performance optimale. Les algorithmes de classification les plus performants sont l'AdaBoost et le NeuralNet. Les résultats ont montré que le modèle est beaucoup plus performant avec 6 attributs ad-hoc ou avec 5 attributs par analyse de variance univariée. Les classificateurs AdaBoost et NeuralNet donnent des résultats presque parfaits pour les deux drapeaux.

### English
The presented project required difficult data preprocessing, particularly to isolate the flag in the selected images. The binary mask obtained through data segmentation allowed us to crop the images to get the area most likely to be the flag. Correlation matrices and visualization helped determine the optimal number of attributes for optimal performance. The most performant classification algorithms are AdaBoost and NeuralNet. The results showed that the model is much more performant with 6 ad-hoc attributes or with 5 attributes through univariate variance analysis. AdaBoost and NeuralNet classifiers give almost perfect results for both flags.
