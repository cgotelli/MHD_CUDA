import numpy as np
import os
import re

def extract_indices(filename):
    """Extrae los índices de inicio y fin del bloque de filas a partir del nombre del archivo."""
    match = re.search(r'_(\d+)_(\d+)_02\.npy', filename)
    if match:
        start_index = int(match.group(1))
        end_index = int(match.group(2))
        return start_index, end_index
    return None, None

def main(directory):
    # Obtener la lista de archivos en el directorio
    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    
    # Filtrar archivos que no coincidan con el patrón esperado
    files = [f for f in files if extract_indices(f)[0] is not None]

    # Ordenar los archivos según el índice de inicio del bloque de filas
    files.sort(key=lambda f: extract_indices(f)[0])

    # Crear una lista para almacenar las partes de la matriz
    matrix_parts = []

    # Cargar y agregar las matrices en la lista correspondiente
    for file in files:
        filepath = os.path.join(directory, file)
        print(f"Leyendo y añadiendo a la matriz: {file}")  # Imprime el nombre del archivo
        matrix_part = np.load(filepath).astype(np.float32)  # Convertir a float32 inmediatamente
        matrix_parts.append(matrix_part)

    # Reconstruir la matriz cuadrada original concatenando las partes verticalmente
    final_matrix = np.vstack(matrix_parts)

    # Copiar la parte triangular superior a la triangular inferior bloque por bloque
    n = final_matrix.shape[0]
    block_size = 1000  # Tamaño del bloque, ajustable para equilibrar entre uso de memoria y rendimiento

    for i in range(0, n, block_size):
        for j in range(i + 1, n, block_size):
            i_end = min(i + block_size, n)
            j_end = min(j + block_size, n)
            final_matrix[j:j_end, i:i_end] = final_matrix[i:i_end, j:j_end].T

    # Verificar si la matriz es cuadrada
    if final_matrix.shape[0] == final_matrix.shape[1]:
        # Guardar la matriz reconstruida en el mismo directorio
        output_path = os.path.join(directory, "final_dense_matrix.npy")
        np.save(output_path, final_matrix)
        print(f"Matriz reconstruida guardada en: {output_path}")
    else:
        print("Error: La matriz reconstruida no es cuadrada.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python assemble_dense_matrix.py <directorio>")
    else:
        main(sys.argv[1])
