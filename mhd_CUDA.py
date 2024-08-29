import os
import time
from PIL import Image
import numpy as np
import torch

def load_images_to_points(directory):
    image_files = [f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]
    sets_of_points = []

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = Image.open(image_path).convert('L')  # Convertir a escala de grises
        image_np = np.array(image)
        
        # Obtener coordenadas de los puntos blancos (asumimos un umbral de 128 para considerar píxeles blancos)
        points = np.column_stack(np.where(image_np > 128))
        
        # Filtrar para tomar solo uno de cada dos puntos
        points = points[::2]
        
        sets_of_points.append(points)

    print("Listo con los puntos")
    return sets_of_points

def calculate_mhd_batch(tensor1, tensor2):
    def tensor_to_points(tensor):
        return torch.nonzero(tensor.squeeze() > 0.1, as_tuple=False).float()
    
    A = tensor_to_points(tensor1).cuda()
    B = tensor_to_points(tensor2).cuda()
    if A.size(1) != B.size(1):
        raise ValueError("Both groups of points have different dimensions.")
    
    dist_matrix = torch.cdist(A.unsqueeze(0), B.unsqueeze(0)).squeeze(0)
    fhd = torch.mean(torch.min(dist_matrix, dim=1)[0])
    rhd = torch.mean(torch.min(dist_matrix, dim=0)[0])
    mhd = torch.max(fhd, rhd)
    
    return mhd.item()

def compute_and_save_distance_matrix_block(sets_of_points, block_start, block_size, save_dir, device):
    n = len(sets_of_points)
    end = min(block_start + block_size, n)
    block_height = end - block_start
    
    # Convertir los sets de puntos a tensores y moverlos a la GPU
    tensors = [torch.tensor(points, dtype=torch.float32).to(device) for points in sets_of_points]

    # Crear submatriz de distancias con NaN en la parte inferior triangular
    distance_matrix_block = np.full((block_height, n), np.nan)

    for i in range(block_height):
        tensor1 = tensors[block_start + i]
        distances = []
        for j in range(block_start + i, n):
            tensor2 = tensors[j]
            mhd = calculate_mhd_batch(tensor1, tensor2)
            distances.append(mhd)
        
        distance_matrix_block[i, block_start + i:n] = distances

        # Imprimir mensaje cada cinco líneas procesadas
        if (i + 1) % 5 == 0:
            print(f"Processed lines {block_start + i - 3} to {block_start + i + 1}")

    # Guardar la submatriz de distancias en el disco
    np.save(os.path.join(save_dir, f'distance_matrix_block_{block_start}_{end}_02.npy'), distance_matrix_block)

def main(directory, start_block=0):
    start_time = time.time()

    # Cargar imágenes y convertirlas en conjuntos de puntos
    sets_of_points = load_images_to_points(directory)
    n = len(sets_of_points)
    block_size = 100  # Ajusta el tamaño del bloque según la memoria disponible

    # Crear directorio para guardar los bloques de la matriz de distancias
    save_dir = 'G:\\distance_matrix_blocks'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Seleccionar dispositivo (GPU o CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for block_start in range(start_block, n, block_size):
        compute_and_save_distance_matrix_block(sets_of_points, block_start, block_size, save_dir, device)
        print(f"Processed block {block_start} to {min(block_start + block_size, n)}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Todos los bloques de la matriz de distancias han sido guardados en '{save_dir}'")
    print(f"Tiempo de ejecución: {elapsed_time:.2f} segundos")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Uso: python script.py <directorio_de_imagenes> [<start_block>]")
    else:
        directory = sys.argv[1]
        start_block = int(sys.argv[2]) if len(sys.argv) == 3 else 0
        main(directory, start_block)
