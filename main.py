import ssl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

# Contournement SSL pour le téléchargement du dataset sur macOS
ssl._create_default_https_context = ssl._create_unverified_context

# --- Configuration ---
BATCH_SIZE = 32
EPOCHS = 5
CLASSES = ('avion', 'voiture', 'oiseau', 'chat', 'cerf',
           'chien', 'grenouille', 'cheval', 'bateau', 'camion')

# --- Préparation des données ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


# --- Architecture du Modèle ---
class Net(nn.Module):
    """
    Architecture CNN basique (type LeNet) pour la classification d'images 32x32.
    Extrait les caractéristiques via 2 couches de convolution avant la classification linéaire.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def imshow(img):
    """Dénormalise et affiche une image."""
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# --- Script Principal ---
if __name__ == '__main__':

    # Configuration du matériel (Accélération Apple Silicon si disponible)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    net = Net().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # 1. Entraînement
    print("\n[1/3] Début de l'entraînement...")
    for epoch in range(EPOCHS):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 500 == 499:
                print(f'   [Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 500:.3f}')
                running_loss = 0.0

    print('Entraînement terminé.')

    # 2. Évaluation
    print("\n[2/3] Évaluation sur le jeu de test...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'   Accuracy globale : {100 * correct / total:.1f}%')

    # 3. Visualisation
    print("\n[3/3] Génération des prédictions visuelles...")
    images, labels = next(iter(testloader))
    images_gpu = images.to(device)
    outputs = net(images_gpu)
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.cpu()

    fig = plt.figure(figsize=(10, 4))
    for idx in range(4):
        ax = fig.add_subplot(1, 4, idx + 1, xticks=[], yticks=[])

        # Préparation de l'image pour Matplotlib
        img = images[idx] / 2 + 0.5
        plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))

        pred_name = CLASSES[predicted[idx]]
        true_name = CLASSES[labels[idx]]

        color = "green" if pred_name == true_name else "red"
        ax.set_title(f"Pred: {pred_name}\nTrue: {true_name}", color=color)

    plt.show()