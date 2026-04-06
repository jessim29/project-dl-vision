#️ Vision par Ordinateur : Classification d'images avec PyTorch

## Objectif
Création d'un pipeline de Deep Learning "from scratch" pour classifier des images en 10 catégories (avions, voitures, animaux...). Ce projet démontre l'implémentation d'un Réseau de Neurones Convolutifs (CNN) avec PyTorch.

---

##️ Stack Technique
- **Framework :** PyTorch & Torchvision
- **Optimisation :** MPS (Metal Performance Shaders) pour l'accélération GPU sur Apple Silicon.
- **Visualisation :** Matplotlib
- **Dataset :** CIFAR-10 (60 000 images couleur 32x32)

---

## Architecture du Modèle
Le modèle est inspiré de l'architecture classique LeNet :
1. Deux couches de convolution spatiale (Conv2d) pour l'extraction de caractéristiques.
2. Couches de sous-échantillonnage (MaxPool2d) pour réduire la dimensionnalité.
3. Fonctions d'activation non-linéaires (ReLU).
4. Trois couches denses (Linear / Fully Connected) pour la classification finale.

---

## Résultats
- **Fonction de coût (Loss) :** Entropie Croisée (CrossEntropyLoss)
- **Optimiseur :** SGD (Stochastic Gradient Descent) avec Momentum
- **Précision (Accuracy) :** ~53% sur le jeu de test après 5 époques. Le modèle surpasse largement la baseline aléatoire (10%) démontrant sa capacité à extraire des motifs visuels complexes malgré une architecture volontairement minimaliste.
# project-dl-vision
