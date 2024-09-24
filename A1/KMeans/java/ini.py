import random

def generate_initial_centroids(input_file, output_file, k):
    with open(input_file, 'r') as f:
        lines = f.readlines()

    centroids = random.sample(lines, k)

    with open(output_file, 'w') as f:
        for centroid in centroids:
            f.write(centroid)

# Generate k initial centers of mass
generate_initial_centroids('D:\mie1628\hw\A2\data_points.txt', 'initial_centroids_8.txt', 8)
