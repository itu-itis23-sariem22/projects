import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

def classify_with_eigenfaces(X, labels, mean_face, eigenfaces, M_values, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    accuracies = []

    labels_unique = sorted(list(set(labels)))
    label_to_index = {label: idx for idx, label in enumerate(labels_unique)}
    y_true = np.array([label_to_index[l] for l in labels])

    for M in M_values:
        Phi = eigenfaces[:, :M]
        X_centered = X - mean_face
        X_proj = X_centered @ Phi

        y_pred = []

        for i in range(len(X_proj)):
            xi = X_proj[i]
            X_rest = np.delete(X_proj, i, axis=0)
            y_rest = np.delete(y_true, i)

            dists = np.linalg.norm(X_rest - xi, axis=1)
            nearest = np.argmin(dists)
            predicted_label = y_rest[nearest]
            y_pred.append(predicted_label)

        acc = accuracy_score(y_true, y_pred)
        accuracies.append(acc)
        print(f"Graph M={M}: Accuracy = {acc*100:.2f}%")

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap='Blues', xticks_rotation='vertical')
        plt.title(f"Confusion Matrix (M={M})")
        plt.savefig(os.path.join(output_dir, f"confusion_matrix_M{M}.png"))
        plt.close()

    plt.figure()
    plt.plot(M_values, [a * 100 for a in accuracies], marker='o')
    plt.title("Recognition Accuracy vs Number of Eigenfaces")
    plt.xlabel("Number of Eigenfaces (M)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "accuracy_vs_eigenfaces.png"))
    plt.close()


def compute_pca(X_centered):

    print("Computing covariance matrix...")
    C_small = X_centered @ X_centered.T

    print("Computing eigenvalues and eigenvectors...")
    eigenvalues, eigenvectors = np.linalg.eigh(C_small)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    eigenfaces = X_centered.T @ eigenvectors
    eigenfaces = eigenfaces / np.linalg.norm(eigenfaces, axis=0)

    return eigenfaces, eigenvalues


def save_eigenfaces(eigenfaces, output_dir, num_components=10):
    os.makedirs(output_dir, exist_ok=True)
    for i in range(num_components):
        face = eigenfaces[:, i].reshape(92, 112)
        plt.imshow(face, cmap='gray')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"ef_{i}_10.png"), bbox_inches='tight', pad_inches=0)
        plt.close()


def plot_eigenvalues(eigenvalues, output_path):
    plt.figure()
    plt.plot(eigenvalues)
    plt.title("Eigenvalues (Variance by Component)")
    plt.xlabel("Component")
    plt.ylabel("Eigenvalue")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "eigenvalues.png"))
    plt.close()


def plot_cumulative_variance(eigenvalues, output_path):
    cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
    plt.figure()
    plt.plot(cumulative)
    plt.title("Cumulative Variance Explained")
    plt.xlabel("Number of Components")
    plt.ylabel("Total Variance")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, "cumulative_variance.png"))
    plt.close()


def load_images(data_path, img_size=(112, 92)):
    images = []
    labels = []
    for person in sorted(os.listdir(data_path)):
        person_dir = os.path.join(data_path, person)
        if os.path.isdir(person_dir):
            for img_name in sorted(os.listdir(person_dir)):
                img_path = os.path.join(person_dir, img_name)
                img = Image.open(img_path).resize(img_size).convert('L')  # Convert to grayscale
                img_np = np.asarray(img).flatten()  # Vectorize image
                images.append(img_np)
                labels.append(person)
    return np.array(images), np.array(labels)

def reconstruct_and_compare(X, mean_face, eigenfaces, labels, person_ids, M_values, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    mse_report = []

    for person_id in person_ids:
        indices = np.where(labels == f"s{person_id}")[0]
        if len(indices) == 0:
            print(f"Person s{person_id} not found!")
            continue
        idx = indices[0]
        x = X[idx]
        x_centered = x - mean_face

        for M in M_values:
            Phi = eigenfaces[:, :M]
            weights = Phi.T @ x_centered
            x_reconstructed = Phi @ weights + mean_face

            mse = np.mean((x - x_reconstructed) ** 2)
            mse_report.append(f"s{person_id}, M={M}, MSE={mse:.2f}")
            print(f"s{person_id}, M={M} → MSE: {mse:.2f}")

            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(x.reshape(92, 112), cmap='gray')
            axes[0].set_title("Original")
            axes[0].axis('off')

            axes[1].imshow(x_reconstructed.reshape(92, 112), cmap='gray')
            axes[1].set_title(f"Reconstructed (M={M})")
            axes[1].axis('off')

            filename = f"comparison_{person_id}_M{M}.png"
            fig.tight_layout()
            plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', pad_inches=0)
            plt.close()

    with open(os.path.join(output_dir, "mse_reconstruction.txt"), "w") as f:
        f.write("\n".join(mse_report))


def compute_mean_face(images):
    return np.mean(images, axis=0)

def save_face_image(face_vector, filename):
    plt.imshow(face_vector.reshape(92, 112), cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)
    plt.close()

def add_gaussian_noise(image, std):
    noise = np.random.normal(0, std, image.shape)
    noisy = image + noise * 255
    return np.clip(noisy, 0, 255)

def add_salt_and_pepper_noise(image, amount):
    noisy = image.copy()
    n_pixels = int(amount * image.size)
    coords = np.random.randint(0, image.size, n_pixels)
    salt_pepper = np.random.choice([0, 255], size=n_pixels)
    flat = noisy.flatten()
    flat[coords] = salt_pepper
    return flat.reshape(image.shape)

def test_noise_robustness(X, labels, mean_face, eigenfaces,
                          person_ids, M, noise_levels, output_dir, noise_type='gaussian'):
    os.makedirs(output_dir, exist_ok=True)
    Phi = eigenfaces[:, :M]
    X_centered = X - mean_face
    X_proj_clean = X_centered @ Phi

    label_to_index = {l: i for i, l in enumerate(sorted(set(labels)))}
    y_true = np.array([label_to_index[l] for l in labels])

    test_indices = [np.where(labels == f"s{pid}")[0][0] for pid in person_ids]
    test_faces = X[test_indices]
    test_labels = y_true[test_indices]

    accuracies = []

    for noise_level in noise_levels:
        y_pred = []

        for i, face in enumerate(test_faces):
            if noise_type == 'gaussian':
                noisy = add_gaussian_noise(face, noise_level)
            else:
                noisy = add_salt_and_pepper_noise(face, noise_level)

            noisy_centered = noisy - mean_face
            noisy_proj = noisy_centered @ Phi

            dists = np.linalg.norm(X_proj_clean - noisy_proj, axis=1)
            nearest_idx = np.argmin(dists)
            pred_label = y_true[nearest_idx]
            y_pred.append(pred_label)

            out_img = noisy.reshape(92, 112)
            fname = f"noisy_{noise_type}_{str(noise_level).replace('.', '')}_{person_ids[i]}.png"
            plt.imshow(out_img, cmap='gray')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', pad_inches=0)
            plt.close()

        acc = accuracy_score(test_labels, y_pred)
        accuracies.append(acc)
        print(f"{noise_type.title()} Noise {noise_level}: Accuracy = {acc*100:.2f}%")

    plt.figure()
    plt.plot(noise_levels, [a * 100 for a in accuracies], marker='o')
    plt.title(f"Accuracy vs {noise_type.title()} Noise Level")
    plt.xlabel("Noise Level")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"accuracy_vs_{noise_type}_noise.png"))
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True, help="Path to ORL face dataset")
    parser.add_argument("--output_path", required=True, help="Where to save output images")
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)

    X, labels = load_images(args.data_path)

    mean_face_all = compute_mean_face(X)
    save_face_image(mean_face_all, os.path.join(args.output_path, "mean_face_all.png"))

    mean_face_subset = compute_mean_face(X[:10])
    save_face_image(mean_face_subset, os.path.join(args.output_path, "mean_face_subset.png"))

    X_centered = X - mean_face_all

    print("Task 1: Mean faces saved and data normalized.")

    print("Starting PCA and Eigenfaces...")

    eigenfaces, eigenvalues = compute_pca(X_centered)

    eigenfaces_dir = os.path.join(args.output_path, "eigenfaces")
    save_eigenfaces(eigenfaces, eigenfaces_dir, num_components=10)

    plot_eigenvalues(eigenvalues, args.output_path)
    plot_cumulative_variance(eigenvalues, args.output_path)

    print("Task 2: Eigenfaces and plots saved.")

    print("Starting face reconstruction...")

    reconstruct_and_compare(
        X=X,
        mean_face=mean_face_all,
        eigenfaces=eigenfaces,
        labels=labels,
        person_ids=[10, 11],
        M_values=[10, 20, 50, 100, 200, 300],
        output_dir=os.path.join(args.output_path, "reconstructed")
    )

    print("Task 3: Faces reconstructed and MSE saved.")

    print("Starting face classification...")

    classify_with_eigenfaces(
        X=X,
        labels=labels,
        mean_face=mean_face_all,
        eigenfaces=eigenfaces,
        M_values=[10, 20, 50, 100, 200, 300],
        output_dir=os.path.join(args.output_path, "recognition")
    )

    print("Task 4: Classification done and confusion matrices saved.")

    print("Starting noise robustness evaluation...")

    noise_levels = [0.1, 0.2, 0.3, 0.4, 0.5]
    selected_people = list(range(1, 11))  # s1 to s10

    test_noise_robustness(
        X=X,
        labels=labels,
        mean_face=mean_face_all,
        eigenfaces=eigenfaces,
        person_ids=selected_people,
        M=50,
        noise_levels=noise_levels,
        output_dir=os.path.join(args.output_path, "noise", "gaussian"),
        noise_type='gaussian'
    )

    test_noise_robustness(
        X=X,
        labels=labels,
        mean_face=mean_face_all,
        eigenfaces=eigenfaces,
        person_ids=selected_people,
        M=50,
        noise_levels=noise_levels,
        output_dir=os.path.join(args.output_path, "noise", "saltpepper"),
        noise_type='salt'
    )

    print("Task 5: Noise tests and accuracy plots saved.")




