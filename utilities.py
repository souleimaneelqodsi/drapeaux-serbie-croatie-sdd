from typing import Any, Dict, Union
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from PIL import Image, ExifTags  # type: ignore
from sklearn.model_selection import StratifiedShuffleSplit  # type: ignore
import sys

def crop_image(img: Image.Image, crop_size: int = 32) -> Image.Image:
    """Crop a PIL image to crop_size x crop_size."""
    # First determine the bounding box
    width, height = img.size
    new_width = 1.0 * crop_size
    new_height = 1.0 * crop_size
    left = round((width - new_width) / 2)
    top = round((height - new_height) / 2)
    right = round((width + new_width) / 2)
    bottom = round((height + new_height) / 2)
    # Then crop the image to that box
    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img


def image_to_vector(img: Image.Image) -> np.ndarray:
    """Return the values of all the pixels of the Image as a vector"""
    return np.array(img).ravel()


def image_to_series(img: Image.Image) -> np.ndarray:
    """Return the values of all the pixels of the Image as a series"""
    # Note: the transparency layer is ignored.
    return pd.Series(np.array(img)[:, :, :3].ravel())


def df_cross_validate(df, sklearn_model, sklearn_metric, n=10, verbose=False):
    """Compute the performance and error bars of a classifier by cross validation

    Input:
    - a data table whose last column contains the ground truth
    - a Scikit-learn model; or anything that quacks like it
    - a Scikit-learn performance metric;
      it can either be an loss (error rate) or an accuracy (success rate).
    """
    X = df.iloc[:, :-1].to_numpy()
    Y = df.iloc[:, -1].to_numpy()
    SSS = StratifiedShuffleSplit(n_splits=n, test_size=0.5, random_state=5)
    Perf_tr = np.zeros([n, 1])
    Perf_te = np.zeros([n, 1])
    k = 0
    for train_index, test_index in SSS.split(X, Y):
        if verbose:
            print("TRAIN:", train_index, "TEST:", test_index)
        Xtrain, Xtest = X[train_index], X[test_index]
        Ytrain, Ytest = Y[train_index], Y[test_index]
        sklearn_model.fit(Xtrain, Ytrain.ravel())
        Ytrain_predicted = sklearn_model.predict(Xtrain)
        perf_tr = sklearn_metric(Ytrain.ravel(), Ytrain_predicted.ravel())
        # if verbose:
        #    print("TRAINING PERFORMANCE METRIC", perf_tr)
        Perf_tr[k] = perf_tr
        Ytest_predicted = sklearn_model.predict(Xtest)
        perf_te = sklearn_metric(Ytest.ravel(), Ytest_predicted.ravel())
        # if verbose:
        #    print("TEST PERFORMANCE METRIC:", perf_te)
        Perf_te[k] = perf_te
        k = k + 1

    perf_tr_ave = np.mean(Perf_tr)
    perf_te_ave = np.mean(Perf_te)
    sigma_tr = np.std(Perf_tr)
    sigma_te = np.std(Perf_te)
    if verbose:
        metric_name = sklearn_metric.__name__.upper()
        print(
            "*** AVERAGE TRAINING {0:s} +- STD: {1:.2f} +- {2:.2f} ***".format(
                metric_name, perf_tr_ave, sigma_tr
            )
        )
        print(
            "*** AVERAGE TEST {0:s} +- STD: {1:.2f} +- {2:.2f} ***".format(
                metric_name, perf_te_ave, sigma_te
            )
        )
    return (perf_tr_ave, sigma_tr, perf_te_ave, sigma_te)


def check_na(df):
    """Finds whether there are missing values."""
    # I used this post
    # https://dzone.com/articles/pandas-find-rows-where-columnfield-is-null
    na_columns = df.columns[df.isna().any()]
    found_na = df[df.isna().any(axis=1)][na_columns]
    print(found_na.head())
    return found_na


def standardize_df(df):
    """Standardize all the columns except the last one (target values)."""
    df_scaled = (df - df.mean()) / df.std()
    df_scaled.iloc[:, -1] = df.iloc[:, -1]
    return df_scaled


def difference_filter(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Extract a numpy array D = R-(G+B)/2 from a PIL image."""
    M = np.array(img)
    R = 1.0 * M[:, :, 0]
    G = 1.0 * M[:, :, 1]
    B = 1.0 * M[:, :, 2]
    D = R - (G + B) / 2
    return D


def value_filter(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    """Extract a numpy array V = (R+G+B)/3 from a PIL image."""
    M = np.array(img)
    R = 1.0 * M[:, :, 0]
    G = 1.0 * M[:, :, 1]
    B = 1.0 * M[:, :, 2]
    V = (R + G + B) / 3
    return V


def foreground_filter(
    img: Union[Image.Image, np.ndarray], theta: int = 150
) -> np.ndarray:
    """Create a black and white image outlining the foreground."""
    M = np.array(img)  # In case this is not yet a Numpy array
    G = np.min(M[:, :, 0:3], axis=2)
    F = G < theta
    return F


def foreground_redness_filter(
    img: Union[Image.Image, np.ndarray], theta: float = 2 / 3
) -> np.ndarray:
    """Extract a numpy array with True as foreground
    and False as background from a PIL image.
    Parameter theta is a relative binarization threshold."""
    D = difference_filter(img)
    V = value_filter(img)
    F0 = np.maximum(D, V)
    threshold = theta * (np.max(F0) - np.min(F0))
    F = F0 > threshold
    return F

def foreground_redness_filter(
    img: Union[Image.Image, np.ndarray], theta: float = 2 / 3
) -> np.ndarray:
    """Extract a numpy array with True as foreground
    and False as background from a PIL image.
    Parameter theta is a relative binarization threshold."""
    D = difference_filter(img)
    V = value_filter(img)
    F0 = np.maximum(D, V)
    threshold = theta * (np.max(F0) - np.min(F0))
    F = F0 > threshold
    return F


def invert_if_light_background(img: np.ndarray) -> np.ndarray:
    """Create a black and white image outlining the foreground."""
    if np.count_nonzero(img) > np.count_nonzero(np.invert(img)):
        return np.invert(img)
    else:
        return img


def crop_around_center(img, center, crop_size=32, verbose=False):
    """Crop a PIL image to crop_size x crop_size."""
    # First determine the center of gravity
    x0, y0 = center[1], center[0]
    # Initialize the image with the old_style cropped image
    # cropped_img = crop_image(img)
    if verbose:
        print(x0, y0)
    # Determine the bounding box
    width, height = img.size
    if verbose:
        print(width, height)
    new_width = 1.0 * crop_size
    new_height = 1.0 * crop_size
    if verbose:
        print(new_width, new_height)
    left = round(x0 - new_width / 2)
    top = round(y0 - new_height / 2)
    right = round(x0 + new_width / 2)
    bottom = round(y0 + new_height / 2)
    if verbose:
        print(left, top, right, bottom)
    return img.crop((left, top, right, bottom))


def convert_image(img: Image.Image, verbose=True) -> np.ndarray:
    """Return a vector of flattened pixel values of a cropped image."""
    return np.array(img).ravel()


def transparent_background(
    img: Union[Image.Image, np.ndarray], foreground: np.ndarray
) -> Image.Image:
    """Return a copy of the image with background set to transparent.

    Which pixels belong to the background is specified by
    `foreground`, a 2D array of booleans of the same size as the
    image.
    """
    M = np.array(img)
    M[~foreground] = [0, 0, 0]
    return Image.fromarray(np.insert(M, 3, foreground * 255, axis=2))


def contours(img: Image.Image) -> np.ndarray:
    M = np.array(img)
    contour_horizontal = np.abs(M[:-1, 1:] ^ M[:-1, 0:-1])
    contour_vertical = np.abs(M[1:, :-1] ^ M[0:-1, :-1])
    return contour_horizontal + contour_vertical


def redness_filter(img: Image.Image) -> float:
    """Return the redness of a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    R = M[:, :, 0] * 1.0
    G = M[:, :, 1] * 1.0
    return R - G


def redness(img: Image.Image) -> float:
    """Return the redness of a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    R = M[:, :, 0] * 1.0
    G = M[:, :, 1] * 1.0
    F = M[:, :, 3] > 0
    return np.mean(R[F]) - np.mean(G[F])


def greenness(img: Image.Image) -> float:
    """Return the greenness of a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    G = M[:, :, 1] * 1.0
    B = M[:, :, 2] * 1.0
    F = M[:, :, 3] > 0
    return np.mean(G[F]) - np.mean(B[F])


def blueness(img: Image.Image) -> float:
    """Return the blueness of a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    B = M[:, :, 2] * 1.0
    R = M[:, :, 0] * 1.0
    F = M[:, :, 3] > 0
    return np.mean(B[F]) - np.mean(R[F])


def elongation(img: Image.Image) -> float:
    """Extract the scalar value elongation from a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    F = M[:, :, 3] > 0
    # Build the cloud of points given by the foreground image pixels
    xy = np.argwhere(F)
    if len(xy) <= 5:
        # Not enough points to compute a meaningful elongation
        return np.NaN
    # Center the data
    C = np.mean(xy, axis=0)
    Cxy = xy - np.tile(C, [xy.shape[0], 1])
    # Apply singular value decomposition
    U, s, V = np.linalg.svd(Cxy)
    return s[0] / s[1]


def perimeter(img: Image.Image) -> float:
    """Extract the scalar value perimeter from a PIL image."""
    C = contours(img)
    return np.count_nonzero(C)


def surface(img: Image.Image) -> float:
    """Extract the scalar value surface from a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    F = M[:, :, 3] > 0
    return np.count_nonzero(F)


def get_colors(img):
    """Extract various colors from a PIL image."""
    M = np.array(img)
    F = M[:, :, 3] > 0
    R = 1.0 * M[:, :, 0]
    G = 1.0 * M[:, :, 1]
    B = 1.0 * M[:, :, 2]
    Mx = np.maximum(np.maximum(R, G), B)
    Mn = np.minimum(np.minimum(R, G), B)
    C = Mx - Mn  # Chroma
    D1 = R - (G + B) / 2
    D2 = G - B
    D3 = G - (R + B) / 2
    D4 = B - R
    D5 = B - (G + R) / 2
    D6 = R - G
    # Hue
    # H1 = np.divide(D2, C)
    # H2 = np.divide(D4, C)
    # H3 = np.divide(D6, C)
    # Luminosity
    V = (R + G + B) / 3
    # Saturation
    # S = np.divide(C, V)
    # We avoid divisions so we don't get divisions by 0
    # Now compute the color features
    if not F.any():
        return pd.Series(dtype=float)
    c = np.mean(C[F])
    return pd.Series({
        "R": np.mean(R[F]),
        "G": np.mean(G[F]),
        "B": np.mean(B[F]),
        "M": np.mean(Mx[F]),
        "m": np.mean(Mn[F]),
        "C": np.mean(C[F]),
        "R-(G+B)/2": np.mean(D1[F]),
        "G-B": np.mean(D2[F]),
        "G-(R+B)/2": np.mean(D3[F]),
        "B-R": np.mean(D4[F]),
        "B-(G+R)/2": np.mean(D5[F]),
        "R-G": np.mean(D6[F]),
        "(G-B)/C": np.mean(D2[F]) / c,
        "(B-R)/C": np.mean(D4[F]) / c,
        "(R-G)/C": np.mean(D5[F]) / c,
        "(R+G+B)/3": np.mean(V[F]),
        "C/V": c / np.mean(V[F]),
        })


def feature_learning_curve(data_df, sklearn_model, sklearn_metric):
    """Run cross-validation on nested subsets of features
    generated by the Pearson correlated coefficient."""
    print(sklearn_model.__class__.__name__.upper())
    corr = data_df.corr()
    sval = corr["class"][:-1].abs().sort_values(ascending=False)
    ranked_columns = sval.index.values
    print(ranked_columns)
    result_df = pd.DataFrame(columns=["perf_tr", "std_tr", "perf_te", "std_te"])
    for k in range(len(ranked_columns)):
        df = data_df[np.append(ranked_columns[0 : k + 1], "class")]
        rdf = pd.DataFrame(
            data=[df_cross_validate(df, sklearn_model, sklearn_metric)],
            columns=["perf_tr", "std_tr", "perf_te", "std_te"],
        )
        result_df = pd.concat([result_df, rdf], ignore_index=True)
    return result_df, ranked_columns


def systematic_model_experiment(data_df, model_name, model_list, sklearn_metric):
    """Run cross-validation on a bunch of models and collect the results."""
    result_df = pd.DataFrame(columns=["perf_tr", "std_tr", "perf_te", "std_te"])
    for name, model in zip(model_name, model_list):
        result_df.loc[name] = df_cross_validate(data_df, model, sklearn_metric)
    return result_df


def highlight_above_median(s):
    """Highlight values in a series above their median."""
    medval = s.median()
    return ["background-color: cyan" if v > medval else "" for v in s]


def analyze_model_experiments(result_df):
    tebad = result_df.perf_te < result_df.perf_te.median()
    trbad = result_df.perf_tr < result_df.perf_tr.median()
    overfitted = tebad & ~trbad
    underfitted = tebad & trbad
    result_df["Overfitted"] = overfitted
    result_df["Underfitted"] = underfitted
    return result_df.style.apply(highlight_above_median)


def extract_metadata(img: Image.Image) -> pd.Series:
    return pd.Series({ExifTags.TAGS[key]: value
                      for key, value in img.getexif().items()},
                     dtype=object)


### Nos fonctions :

def colorStrongFilter(img : Image.Image, c : str, i1 : int, i2 : int) -> np.ndarray:
    """
    Fonction qui permet de mettre en évidence l'intensité d'une couleur (r, v ou b) dans une image en établissant deux seuils : un permettant d'évaluer la force de la couleur dans chaque pixel et l'autre la faiblesse des deux autres couleurs dans chaque pixel.
    
    Arguments :
    -----------
        (Image.Image) : l'image sur laquelle effectuer le traitement
        (str) : couleur ('R', 'G', 'B')
        (int) : seuil de force de la couleur choisie
        (int) : seuil de faiblesse des deux autres composantes couleur
    
    Returns :
    ---------
        (np.ndarray) : tableau de booléens (True quand les seuils sont respectés, False quand ils ne le sont pas)
    """
    M = np.array(img)
    RGB = {'R': 0, 'G' : 1, 'B' : 2}
    cond1 = M[:, :, RGB[c]] > i1  # quels pixels sont forts dans la couleur cherchée ? 
    del RGB[c] # on veut continuer à travailler seulement sur les deux couleurs restantes
    moy = (M[:, :, list(RGB.values())[0]] + M[:, :, list(RGB.values())[1]])/2 # on fait une moyenne des deux
    cond2 = moy < i2 # on cherche les pixels faibles en ces deux couleurs (niveau de faiblesse recherchée défini par le paramètre i2)
    return np.logical_and(cond1, cond2) # on cherche les pixels satisfaisant les deux conditions à la fois
  
def whiteness(img: Image.Image, seuil: int = 190) -> np.ndarray:
    """
    Calcule un masque de pixels blancs pour une image donnée, en utilisant une méthode de seuillage.

    Args:
        img (PIL.Image.Image): L'image à traiter.
        seuil (int): La valeur de seuil pour le seuillage. Tous les pixels dont la valeur est supérieure à ce seuil seront considérés comme blancs.

    Returns:
        np.ndarray: Un masque de pixels blancs, représenté par un tableau numpy de dimensions (hauteur, largeur) avec des valeurs booléennes.
        
    """
    # Conversion de l'image en un tableau numpy
    M = np.array(img)
    # Création d'un tableau de seuil avec la même taille que l'image
    threshold = np.full(M.shape, seuil, dtype=M.dtype)
    return np.all(M > threshold, axis=2)

def axisCheck(img: np.ndarray, y: int, x: int) -> tuple:
    """
    Fonction qui compte le nombre de pixels blancs ("True") sur les axes verticaux et horizontaux passant par un pixel donné d'une image binaire.

    Arguments :
    -----------
        img : np.ndarray
            Image binaire filtrée pour éliminer le bruit de fond et ne garder que l'objet d'intérêt.
        y, x : int
            Coordonnées du pixel à partir duquel tracer les axes verticaux et horizontaux.

    Returns :
    ---------
        trueCountY : int 
            Nombre de pixels blancs ("True") sur l'axe vertical passant par le pixel de coordonnées (y, x).
        trueCountX : int 
            Nombre de pixels blancs ("True") sur l'axe horizontal passant par le pixel de coordonnées (y, x).
    """
    M = np.array(img)
    dim = M.shape
    trueCountX = 0
    trueCountY = sum(1 for i in range(dim[0]) if M[i, x])
    trueCountX = sum(bool(i) for i in M[y])

    return (trueCountY, trueCountX)


def centerFinder(img: Image.Image, center : int, precision: int = 10) -> int:
    """
    Cherche la zone a conserver lors du troncage a partir d'un pixel qui lui sert de centre.

    Arguments:
    ----------
    img : Image.Image
        Image en binaire
    center : int
        le centre de l'image 
    precision : int, optional
        indique la precision du troncage, utile pour déterminer les extremités de la zone à conserver lors du troncage

    Returns:
    --------
    radius : int
        Le rayon de la zone qu'on va conserver lors du troncage
    """
# On définit une zone qui va conserver le drapeau lors du troncage
    M = np.array(img)
    crop1 = [-1, -1] 
    crop2 = [-1, -1]
    dim = M.shape

    for y in range(dim[0]) : 
        for x in range(dim[1]) :
            result = axisCheck(M, y, x)
            if result[0] >= precision: #Résultat sur l'axe vertical
                crop1[0] = y
            if result[1] >= precision :
                crop1[1] = x
            if -1 not in crop2 :
                break
    for y in range(dim[0] - 1, 0, -1) : 
        for x in range(dim[1] - 1, 0, -1) :
            result = axisCheck(M, y, x)
            if result[0] >= precision: #Résultat sur l'axe vertical
                crop2[0] = y
            if result[1] >= precision :
                crop2[1] = x
            if -1 not in crop2 : 
                break           
    if -1 in crop2 or -1 in crop1 :  #L'image est vide / on n'a pas réussi à définir une boîte
        return 0
    # Les 2 pixels qu'on a trouvé vont définir l'air de la boîte
    radius = max([crop2[1] - crop1[1], crop2[0] - crop1[0]]) // 2
    radius -= ((center[1] + radius) % (dim[1] - 1)) + ((center[0] + radius) % (dim[0] - 1)) #Vérifie si le rayon de la boîte ne va dépasser l'image en partant du pointer "center".
    if center[1] - radius < 0 :
        radius += abs(center[1] - radius)
    if center[0] - radius < 0 :
        radius += abs(center[0] - radius)
    
    return abs(radius) // 1



def blockCreation(img : Image.Image, yKernel : int, xKernel : int, blockSize : int) -> list:
    """Fonction qui crée un bloc de taille blockSize autour d'un pixel noyau, et le stocke dans un tableau
    
    Arguments :
    -----------
        (Image.Image) : image
        (int) : coordonéee en abscisse du noyau
        (int) : coordonnée en ordonnée du noyau
        (int) : taille du bloc souhaitée
    
    
    Returns :
    ---------
        (list) : bloc de taille blockSize * blockSize centré autour du kernel
    """
    M = np.array(img)
    block = []
    rayonBlock = blockSize//2
    for y in range(yKernel - rayonBlock, yKernel + rayonBlock + 1): 
        for x in range(xKernel - rayonBlock, xKernel + rayonBlock + 1): 
            if y < 0: 
                break
            if x < 0:
                continue
            if x >= M.shape[1]:
                break
            if y >= M.shape[0] :
                break
            block.append(M[y , x])
    return block


def appartenance(img : Image.Image, y : int, x : int, blockSize : int, rougeIntensite : int, bleuIntensite : int, coeffRouge : float, coeffBleu : float) -> np.ndarray:
    """Selon les données trouvées dans le bloc généré par blockCreation, la fonction permet de déterminer si le pixel entré en paramètres appartient au drapeau ou non à travers un système de score
    
    Arguments :
    -----------
        (Image.Image) : image
        (int) : coordonée en abscisse du pixel
        (int) : coordonnée en ordonnée du pixel
        (int) : taille du bloc utilisée pour effectuer la recherche (influe sur la performance de la fonction)
        (int) : intensité de rougueur recherchée dans le bloc
        (int) intensité de bleuissement recherchée dans le bloc
        (float) : coefficient i permettant de déterminer si le rouge est i-ème fois plus grand que le vert et le bleu
        (float) : coefficient i permettant de déterminer si le bleu est i-ème fois plus grand que le rouge et le vert
    
    Returns :
    ---------
        (bool) : booléen valant True quand le pixel a un bon score, False sinon
        """
    block = blockCreation(img, y, x, blockSize)
    score = 0
    for i in block: 
        if i[0] > rougeIntensite:
            operation1 = int(i[2] * coeffRouge)
            operation2 = int(i[1] * coeffRouge)
            score += 1 if operation1 < i[0] and operation2 < i[0] else 0.2
        if i[2] > bleuIntensite: 
            operation1 = int(i[0] * coeffBleu)
            operation2 = int(i[1] * coeffBleu)
            score += 1 if operation1 < i[2] and operation2 < i[2] else 0.2
    score /= len(block) + sys.float_info.epsilon
    return score >= 0.5

#s = 1

def drapeau(img : Image.Image, rougeIntensite : int = 80, bleuIntensite : int = 80, coeffRouge : float = 1.5, coeffBleu : float = 1.25, seuil : int = None) -> np.ndarray: 
    """
    Fonction renvoyant un tableau de booléens valant True si le pixel associé appartient à un drapeau serbe/croate, False sinon, à partir des données de rougeur, bleuissement et blancheur du drapeau fournies par les fonctions précédemment écrites -> masque binaire de l'image via segmentation des données par seuillage
    
    Arguments :
    -----------
        (Image.Image) : l'image
        (int) : intensité de rouge fournie à la fonction appartenance
        (int) : intensité de bleu fournie à la fonction appartenance
        (float) : coefficient de rouge fourni à la fonction appartenance
        (float) : coefficient de bleu fourni à la fonction appartenance
        (int) : seuil de différence d'intensité fourni à la fonction whiteness

    Returns :
    ---------
        (np.ndarray) : tableau de booléens valant True si le pixel associé appartient à un drapeau serbe/croate, False sinon
    """
    #global s
    M = np.array(img)
    dim = M.shape
    isDrapeau = np.zeros((dim[0], dim[1]))
    blockSize = 3
    for y in range(dim[0]) :
        for x in range(dim[1]) : 
            isDrapeau[y, x] = appartenance(img, y, x, blockSize, rougeIntensite, bleuIntensite, coeffRouge, coeffBleu)
    #print(f"Chargement en cours [{s}/{length}]")
    #s+=1
    print("Opération effectuée")
    return np.logical_or(np.logical_or(isDrapeau, whiteness(img)), yellowness_mask(img))

def crop_flag(img : Image.Image, seuil : float =0.1) -> Image.Image:
    """
    Recadrer la zone masquée où il y a une forte concentration de valeurs vraies dans le masque.

    Paramètres:
        img (numpy.ndarray): l'image d'entrée.
        mask (numpy.ndarray): le masque binaire.
        seuil (float): la proportion minimale de valeurs vraies requise pour être considérée comme une forte concentration.

    Retours:
        numpy.ndarray: l'image recadrée.
    """
    masque = drapeau(img)

    # Calcule la proportion des pixels valant True
    somme_lig = np.sum(masque, axis=1)
    somme_col = np.sum(masque, axis=0)
    lig_props = somme_lig / masque.shape[1]
    col_props = somme_col / masque.shape[0]

    # Trouve les indices des pixels se trouvant dans les zones à forte concentration de valeurs vraies
    row_indices = np.where(lig_props >= seuil)[0]
    col_indices = np.where(col_props >= seuil)[0]

    # Cherche les "frontières" de la zone à rogner
    haut = row_indices.min()
    bas = row_indices.max()
    gauche = col_indices.min()
    droite = col_indices.max()

    return img.crop((gauche, haut, droite, bas))


def yellowness_mask(img : Image.Image, seuil_sup : int = 150, seuil_inf : int = 90) -> np.ndarray:
    """
    Fonction crééant un masque binaire des pixels jaunes, via un procédé de segmentation des données par seuillage et réunion par opération binaire ET
    
    Args:
    ------
        (Image.Image) : image
        (int) : threshold de supériorité pour récupérer les pixels verts et rouges
        (int) : threshold d'infériorité pour séparer les pixels jaunes des pixels blancs
    Returns:
    --------
        (np.ndarray) : tableau de masque de l'image binarisée
    
    """
    M = np.array(img)
    red = M[:, :, 0] >= seuil_sup
    green = M[:, :, 1] >= seuil_sup
    blue = M[:, :, 2] <= seuil_inf
    return np.logical_and(np.logical_and(red, green), blue)


def yellowness(img: Image.Image) -> float:
    """Return the yellowness of a PIL image.

    Assumption: the background pixels are defined by being transparent.
    """
    M = np.array(img)
    R = M[:, :, 0] * 1.0
    G = M[:, :, 1] * 1.0
    Y = 0.5 * (R + G)
    F = M[:, :, 3] > 0
    return np.mean(Y[F])


def axeCreation(M : np.ndarray, centre : tuple) -> list:
    """
    Fonction qui donne les pixels se trouvant sur l'axe horizontal passant par le centre de l'objet. 
    
    Arguments:
    ----------
    (np.array) : np.array de l'image 
    (float, float) : coordonnées du centre de l'objet
    
    Returns:
    --------
    (list) : liste de pixels de M avec leurs composantes RGB.
    
    """
    centreY = int(centre[1]//1) #Conversion en int
    return [
        M[centreY, x]
        for x in range(M.shape[1])
        if (M[centreY, x, 0], M[centreY, x, 1], M[centreY, x, 2])
        != (255, 255, 255)
    ] 

    
def discriminant(img : Image.Image, centre : tuple, bleuIntensite : int = 80, coeffBleu : float = 1.25) -> list:
    """
    Fonction qui détermine si le pixel se trouvant le plus en bas du drapeau en partant du centre est bleu : si oui, cela veut dire qu'il y a des chances que le drapeau soit croate, car le 
    drapeau serbe contient une bande blanche au lieu de bleu.
    
    Arguments:
    ----------
    (Image.Image) : L´image
    tuple (float, float) : coordonnées du centre de l'objet
    
    Returns:
    --------
    (list) : liste de pixels de M avec leurs composantes RGB.
    """
    M = np.array(img)
    centreX =int(centre[0]//1)
    discriminant = [
        M[y, centreX]
        for y in range(M.shape[0])
        if (M[y, centreX, 0], M[y, centreX, 1], M[y, centreX, 2])
        not in [(255, 255, 255), (0, 0, 0)]
    ]
    print(discriminant[-1])
    return (
        discriminant[-1][2] > bleuIntensite
        and discriminant[-1][1] * coeffBleu < discriminant[-1][2]
        and discriminant[-1][0] * coeffBleu < discriminant[-1][2]
    ) 
            
def analyseCentre(img : Image.Image, centre, bleuIntensite = 80, coeffBleu : float = 1.25, power = 190) -> int:
    """
    Fonction qui détermine la catégorie du drapeau selon ce qui se trouve sur l'axe horizontale traversant le centre du drapeau.
    
    Arguments : 
    -----------
        (Image.Image) : L´image
        (int) : Le centre de l´image 
        (int) : La valeur que doit avoir le bleu d'une composante RBG d'un pixel pour passer le test
        (float) : coefficient qui indique que les couleurs rouges et vertes doivent avoir une valeur égale à moins de 75% du bleu pour passer le test
        (power) : l'intensité de blancheur que doit avoir le pixel pour passer le test
    
    Returns:
    --------
        (Int) : Un score, plus il est élevé plus il est probable que ce soit un drapeau croate.
    """
    #On ne prend pas en compte le rouge, car les 2 drapeaux possèdent du rouge dans leurs ecussons aux centres du drapeau.
    M = np.array(img)
    dim = M.shape
    score = 0
    block = axeCreation(M, centre)
    for i in block: 
        if i[2] > bleuIntensite and i[1] * coeffBleu < i[2] and i[0] * coeffBleu < i[2] : #est bleu
                score -= 1 #Score arbitraire; il se peut que le blason du drapeau serbe, qui est de couleur rouge et blanc, poussent la fonction a donné un résultat trompeur
        if (
            i[1] >= power
            and i[2] >= power
            and i[0] >= power
            and (i[0], i[1], i[2]) != (255, 255, 255)
        ):
            score += 1
    return score