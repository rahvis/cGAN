import argparse
import logging
import torch
import pandas as pd
from data.data_loader import load_iris_data
from utils.metrics import evaluate_metrics
from models.gan import StandardGAN
from models.cgan import ConformalGAN

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def run_gan(X_real, y_real, num_classes, input_dim, device):
    logging.info("Training standard conditional GAN...")
    gan_model = StandardGAN(input_dim=input_dim, noise_dim=10,
                            num_classes=num_classes, device=device)
    synthetic_data, synthetic_labels = gan_model.train(X_real, y_real, epochs=10000, batch_size=32)
    
    # Get GAN metrics (no alpha_values used)
    gan_metrics = evaluate_metrics(X_real, y_real, synthetic_data, synthetic_labels, alpha_values=[])
    
    return gan_metrics

def run_cgan(X_real, y_real, num_classes, input_dim, device):
    logging.info("Training conformalized conditional GAN (C-GAN)...")
    cgan_model = ConformalGAN(input_dim=input_dim, noise_dim=10,
                              num_classes=num_classes, device=device,
                              lambda_reg=0.1, mu_conform=0.1)
    synthetic_data, synthetic_labels = cgan_model.train(X_real, y_real, epochs=10000, batch_size=32)
    
    # Get C-GAN metrics with adaptive alpha values
    cgan_metrics = evaluate_metrics(X_real, y_real, synthetic_data, synthetic_labels, alpha_values=[])
    
    return cgan_metrics

def tabulate_results(gan_metrics, cgan_metrics):
    # Extract GAN results (excluding Interval_Width and Coverage)
    data = {
        'Metric': ['KS_mean', 'Wasserstein_mean', 'Downstream_Accuracy'],
        'GAN': [gan_metrics.get('KS_mean', 'N/A'),
                gan_metrics.get('Wasserstein_mean', 'N/A'),
                gan_metrics.get('Downstream_Accuracy', 'N/A')],
        'C-GAN': [cgan_metrics.get('KS_mean', 'N/A'),
                  cgan_metrics.get('Wasserstein_mean', 'N/A'),
                  cgan_metrics.get('Downstream_Accuracy', 'N/A')]
    }

    # Now dynamically add the Interval_Width metrics for C-GAN
    for alpha in cgan_metrics.get('Interval_Width', {}).keys():
        data['Metric'].append(f'Interval_Width_alpha_{alpha}')
        data['GAN'].append('N/A')  # GAN doesn't have these
        data['C-GAN'].append(cgan_metrics['Interval_Width'].get(alpha, 'N/A'))
    
    # Create a DataFrame to tabulate results
    df = pd.DataFrame(data)
    logging.info("\nEvaluation Metrics Comparison:")
    logging.info("\n" + df.to_string(index=False))

def main(args):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Loading Iris dataset...")
    X_real, y_real = load_iris_data()
    num_classes = len(set(y_real))
    input_dim = X_real.shape[1]
    
    device = get_device()
    logging.info(f"Using device: {device}")
    
    # Run GAN and C-GAN models
    gan_metrics = run_gan(X_real, y_real, num_classes, input_dim, device)
    cgan_metrics = run_cgan(X_real, y_real, num_classes, input_dim, device)
    
    # Tabulate and display the results
    tabulate_results(gan_metrics, cgan_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train conditional GAN or Conformalized GAN on the Iris dataset")
    parser.add_argument("--model", type=str, choices=['gan', 'cgan', 'both'], default="both",
                        help="Select model: 'gan', 'cgan', or 'both'")
    args = parser.parse_args()
    
    # Only run both models if 'both' is selected
    if args.model == 'both':
        main(args)
    elif args.model == 'gan':
        # Add logic to run only GAN if needed
        pass
    elif args.model == 'cgan':
        # Add logic to run only C-GAN if needed
        pass
