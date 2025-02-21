import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import time
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import json
import torch.backends.cudnn as cudnn
import numpy as np

# Import the I2SB model definition
from i2sb_model import I2SBBackbone  # Ensure you have this import from your previous model file

class TestDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_paths = [f for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform or transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.image_paths[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, self.image_paths[idx]

class ModelTester:
    def __init__(self, model_path, device, test_loader, output_dir):
        self.device = device
        self.test_loader = test_loader
        self.output_dir = output_dir
        self.model = self.load_model(model_path)
        os.makedirs(output_dir, exist_ok=True)

    def load_model(self, model_path):
        try:
            # Create model first
            model = I2SBBackbone().to(self.device)
            
            # Load state dict 
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Check if the checkpoint contains the state dict directly or nested
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            model.load_state_dict(state_dict)
            model.eval()
            print(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            raise Exception(f"Error loading model from {model_path}: {str(e)}")

    def run_test(self):
        results = {
            'psnr_values': [],
            'ssim_values': [],
            'processing_times': [],
            'processed_files': []
        }
    
        # Ensure directories exist to save images
        generated_dir = os.path.join(self.output_dir, 'generated_images_2')
        os.makedirs(generated_dir, exist_ok=True)
    
        total_start_time = time.time()
    
        with torch.no_grad():
            # Use half precision to reduce memory usage
            with torch.cuda.amp.autocast():
                for batch, filenames in tqdm(self.test_loader, desc="Testing"):
                    batch = batch.to(self.device)
    
                    # Process batch and measure time
                    start_time = time.time()
                    outputs = self.model(batch)
                    batch_time = time.time() - start_time
    
                    # Iterate over the batch
                    for i in range(len(batch)):
                        original = batch[i].cpu().numpy().transpose(1, 2, 0)  # Convert to numpy (H, W, C)
                        improved = outputs[i].cpu().numpy().transpose(1, 2, 0)  # Convert to numpy (H, W, C)
    
                        # Denormalize images
                        original = (original * 0.5 + 0.5) * 255
                        improved = (improved * 0.5 + 0.5) * 255
    
                        # Clip and convert to uint8
                        original = np.clip(original, 0, 255).astype(np.uint8)
                        improved = np.clip(improved, 0, 255).astype(np.uint8)
    
                        # Save images
                        original_image_path = os.path.join(generated_dir, f"original_{filenames[i]}")
                        improved_image_path = os.path.join(generated_dir, f"generated_{filenames[i]}")
                        Image.fromarray(original).save(original_image_path)
                        Image.fromarray(improved).save(improved_image_path)
    
                        # Calculate PSNR and SSIM
                        psnr_val = psnr(original, improved, data_range=255.0)
                        ssim_val = ssim(original, improved, channel_axis=-1, data_range=255.0)
    
                        results['psnr_values'].append(psnr_val)
                        results['ssim_values'].append(ssim_val)
                        results['processing_times'].append(batch_time * 1000 / len(batch))  # Convert to ms per image
                        results['processed_files'].append(filenames[i])
    
            total_time = time.time() - total_start_time
    
            # Compile final results
            final_results = {
                'average_psnr': float(np.mean(results['psnr_values'])),
                'average_ssim': float(np.mean(results['ssim_values'])),
                'average_processing_time_ms': float(np.mean(results['processing_times'])),
                'total_processing_time_seconds': total_time,
                'total_patches_processed': len(results['processed_files']),
                'min_psnr': float(np.min(results['psnr_values'])),
                'max_psnr': float(np.max(results['psnr_values'])),
                'min_ssim': float(np.min(results['ssim_values'])),
                'max_ssim': float(np.max(results['ssim_values']))
            }
    
            # Save the results to a JSON file
            with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
                json.dump(final_results, f, indent=4)
    
        return final_results

def main():
    # Configuration - adjust based on your specific paths and requirements
    config = {
        'model_path': r"model-flash.pth",  
        'test_data_path': r"patches",  
        'output_dir': 'i2sb_test_results',
        'batch_size': 64  # Adjusted for RTX 3080 Ti
    }

    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Optimization for GPU
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Data loading with optimized transform
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_dataset = TestDataset(config['test_data_path'], transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=8,  # Adjust based on your CPU cores
        pin_memory=True,
        prefetch_factor=2  # Optional: can help with data loading
    )

    # Initialize and run tester
    tester = ModelTester(
        model_path=config['model_path'],
        device=device,
        test_loader=test_loader,
        output_dir=config['output_dir']
    )

    try:
        results = tester.run_test()
        print("\nTest Results:")
        print(f"Average PSNR: {results['average_psnr']:.2f}")
        print(f"Average SSIM: {results['average_ssim']:.2f}")
        print(f"Total processing time: {results['total_processing_time_seconds']:.2f} seconds")
        print(f"Average time per patch: {results['average_processing_time_ms']:.2f} ms")
        print(f"Total patches processed: {results['total_patches_processed']}")
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == '__main__':
    main()