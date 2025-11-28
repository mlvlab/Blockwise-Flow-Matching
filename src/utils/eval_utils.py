from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import re
import os
import scipy
from safetensors.torch import load_file


def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    imgs = sorted(os.listdir(sample_dir), key=lambda x: int(x.split('.')[0]))
    print(len(imgs))
    assert len(imgs) >= num
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{imgs[i]}")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def calculate_inception_stats_cifar(arr, detector_net=None, detector_kwargs=None, batch_size=100, device='cpu'):
    num_samples = arr.shape[0]
    count = 0
    mu = torch.zeros([2048], dtype=torch.float64, device=device)
    sigma = torch.zeros([2048, 2048], dtype=torch.float64, device=device)

    for k in range((arr.shape[0] - 1) // batch_size + 1):
        mic_img = arr[k * batch_size: (k + 1) * batch_size]
        mic_img = torch.tensor(mic_img).permute(0, 3, 1, 2).to(device)
        features = detector_net(mic_img, **detector_kwargs).to(torch.float64)
        if count + mic_img.shape[0] > num_samples:
            remaining_num_samples = num_samples - count
        else:
            remaining_num_samples = mic_img.shape[0]
        mu += features[:remaining_num_samples].sum(0)
        sigma += features[:remaining_num_samples].T @ features[:remaining_num_samples]
        count = count + remaining_num_samples
    assert count == num_samples
    mu /= num_samples
    sigma -= mu.ger(mu) * num_samples
    sigma /= num_samples - 1
    mu = mu.cpu().numpy()
    sigma = sigma.cpu().numpy()
    return mu, sigma

def calculate_inception_stats_imagenet(arr, evaluator, batch_size=100, device='cpu'):
    print("computing sample batch activations...")
    sample_acts = evaluator.read_activations(arr)
    print("computing/reading sample batch statistics...")
    sample_stats, sample_stats_spatial = tuple(evaluator.compute_statistics(x) for x in sample_acts)
    return sample_acts, sample_stats, sample_stats_spatial
    

def compute_fid(mu, sigma, ref_mu=None, ref_sigma=None):
    m = np.square(mu - ref_mu).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, ref_sigma), disp=False)
    fid = m + np.trace(sigma + ref_sigma - s * 2)
    fid = float(np.real(fid))
    return fid