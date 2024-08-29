import os
import time
from PIL import Image
import numpy as np
import torch
from skimage.metrics import hausdorff_distance

def load_images_to_points(directory):
    image_files = [f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.jpg')]
    sets_of_points = []

    for image_file in image_files:
        image_path = os.path.join(directory, image_file)
        image = Image.open(image_path).convert('L')  # Convertir a escala de grises
        image_np = np.array(image)
        
        # Obtener coordenadas de los puntos blancos (asumimos un umbral de 128 para considerar píxeles blancos)
        points = np.column_stack(np.where(image_np > 128))
        sets_of_points.append(points)

    print("Listo con los puntos")
    return sets_of_points

def calculate_mhd(tensor1, tensor2):
    def tensor_to_points(tensor):
        return torch.nonzero(tensor.squeeze() > 0.1, as_tuple=False).float()

    A = tensor_to_points(tensor1)
    B = tensor_to_points(tensor2)
    if A.size(1) != B.size(1):
        raise ValueError("Both groups of points have different dimensions.")
    
    fhd = torch.mean(torch.min(torch.cdist(A, B), dim=1)[0])
    rhd = torch.mean(torch.min(torch.cdist(B, A), dim=1)[0])
    mhd = torch.max(fhd, rhd)
    # print(f"d_A_to_B: {fhd.item()}, d_B_to_A: {rhd.item()}, MHD: {mhd.item()}")

    return mhd.item()

def compute_and_save_distance_matrix_block(sets_of_points, block_start, block_size, save_dir):
    n = len(sets_of_points)
    end = min(block_start + block_size, n)
    block_height = end - block_start
    
    # Crear submatriz de distancias con NaN en la parte inferior triangular
    distance_matrix_block = np.full((block_height, n), np.nan)

    for i in range(block_height):
        for j in range(block_start + i, n):  # Procesar solo la parte triangular superior
            if block_start + i == j:
                dist = 0.0  # La distancia de un punto a sí mismo es cero
            else:
                tensor1 = torch.tensor(sets_of_points[block_start + i], dtype=torch.float32)
                tensor2 = torch.tensor(sets_of_points[j], dtype=torch.float32)
                mhd = calculate_mhd(tensor1, tensor2)
                distance_matrix_block[i, j] = mhd

                # Uncommment if want to compare with skimage's function.
                # tensor1_np = tensor1.numpy()
                # tensor2_np = tensor2.numpy()
                # mhd_skimage = hausdorff_distance(tensor1_np > 0.1, tensor2_np > 0.1, method='modified')
                # print(f"MHD (skimage): {mhd_skimage}")

                # if not np.isclose(mhd, mhd_skimage):
                #     print(f"Discrepancy found at ({block_start + i}, {j}): {mhd} vs {mhd_skimage}")

        # Imprimir mensaje cada dos líneas procesadas
        if (i + 1) % 5 == 0:
            print(f"Processed lines {block_start + i - 3} to {block_start + i +1}")

    # Guardar la submatriz de distancias en el disco
    np.save(os.path.join(save_dir, f'distance_matrix_block_{block_start}_{end}sk.npy'), distance_matrix_block)

def main(directory):
    start_time = time.time()

    # Cargar imágenes y convertirlas en conjuntos de puntos
    sets_of_points = load_images_to_points(directory)
    n = len(sets_of_points)
    block_size = 1000  # Ajusta el tamaño del bloque según la memoria disponible

    # Crear directorio para guardar los bloques de la matriz de distancias
    save_dir = 'G:\\distance_matrix_blocks'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for block_start in range(0, n, block_size):
        compute_and_save_distance_matrix_block(sets_of_points, block_start, block_size, save_dir)
        print(f"Processed block {block_start} to {min(block_start + block_size, n)}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Todos los bloques de la matriz de distancias han sido guardados en '{save_dir}'")
    print(f"Tiempo de ejecución: {elapsed_time:.2f} segundos")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python script.py <directorio_de_imagenes>")
    else:
        directory = sys.argv[1]
        main(directory)
