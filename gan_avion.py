import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

# ==========================================
# 0. DETECCIÓN DE LA TARJETA GRÁFICA (GPU)
# ==========================================
# PyTorch revisa si tienes una GPU NVIDIA compatible. Si no, usa la CPU por defecto.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Entrenando usando: {device.type.upper()} ---")

# ==========================================
# CONFIGURACIÓN Y CATEGORÍA USADA
# ==========================================
CATEGORIA = "airplane"
BATCH_SIZE = 64
LR = 0.0002
EPOCHS = 50
LATENT_DIM = 100 

data = np.load(f'{CATEGORIA}.npy')
data = (data.astype(np.float32) - 127.5) / 127.5
dataset = TensorDataset(torch.tensor(data))
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# ==========================================
# 1. RED NEURONAL GENERADORA
# ==========================================
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM, 256),
            nn.LeakyReLU(0.2), 
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh() 
        )

    def forward(self, z):
        return self.model(z)

# ==========================================
# 2. RED NEURONAL DISCRIMINADORA
# ==========================================
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3), 
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid() 
        )

    def forward(self, img):
        return self.model(img)

# ¡NUEVO! Enviamos las redes neuronales a la memoria de la tarjeta gráfica
generador = Generator().to(device)
discriminador = Discriminator().to(device)

# ==========================================
# 3. FUNCIONES DE PÉRDIDA Y OPTIMIZADORES
# ==========================================
criterion = nn.BCELoss()
optimizer_G = optim.Adam(generador.parameters(), lr=LR)
optimizer_D = optim.Adam(discriminador.parameters(), lr=LR)

# ==========================================
# ENTRENAMIENTO
# ==========================================
for epoch in range(EPOCHS):
    for i, (imgs,) in enumerate(dataloader):
        
        # ¡NUEVO! Enviamos los lotes de imágenes y etiquetas a la gráfica
        imgs = imgs.to(device)
        valid = torch.ones(BATCH_SIZE, 1, device=device)
        fake = torch.zeros(BATCH_SIZE, 1, device=device)

        # --- Entrenar Discriminador ---
        optimizer_D.zero_grad()
        real_pred = discriminador(imgs)
        d_real_loss = criterion(real_pred, valid) 
        
        # ¡NUEVO! El vector de ruido también debe nacer en la gráfica
        z = torch.randn(BATCH_SIZE, LATENT_DIM, device=device)
        gen_imgs = generador(z)
        fake_pred = discriminador(gen_imgs.detach())
        d_fake_loss = criterion(fake_pred, fake) 
        
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # --- Entrenar Generador ---
        optimizer_G.zero_grad()
        g_pred = discriminador(gen_imgs)
        g_loss = criterion(g_pred, valid) 
        
        g_loss.backward()
        optimizer_G.step()

    # ==========================================
    # 4. VER LOS VALORES (Impresión en consola)
    # ==========================================
    if epoch % 10 == 0 or epoch == EPOCHS - 1:
        print(f"Epoch [{epoch}/{EPOCHS}]")
        print(f"  -> Loss Discriminador: {d_loss.item():.4f}")
        print(f"  -> Loss Generador: {g_loss.item():.4f}")
        print(f"  -> Valor Discriminador (Real): {real_pred.mean().item():.4f}")
        print(f"  -> Valor Discriminador (Falso): {fake_pred.mean().item():.4f}\n")

# ==========================================
# 5. VER EL RESULTADO DE LA GAN
# ==========================================
def mostrar_resultados(generador, num_imagenes=5):
    # Generamos ruido en la GPU
    z = torch.randn(num_imagenes, LATENT_DIM, device=device)
    
    # ¡NUEVO! Extraemos los resultados a la CPU para poder dibujarlos con Matplotlib (.cpu())
    dibujos_falsos = generador(z).cpu().detach().numpy()
    
    fig, axes = plt.subplots(1, num_imagenes, figsize=(10, 2))
    fig.suptitle(f"Resultados de la GAN - Categoría: {CATEGORIA.capitalize()}")
    
    for i in range(num_imagenes):
        img = (dibujos_falsos[i].reshape(28, 28) * 127.5) + 127.5
        axes[i].imshow(img, cmap='gray_r')
        axes[i].axis('off')
    plt.show()

mostrar_resultados(generador)
