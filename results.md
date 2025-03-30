# Results

```bash
(venv) (base) rahul@MacBookAir cGAN % python main.py --model both
2025-03-29 17:04:42,016 - INFO - Loading Iris dataset...
2025-03-29 17:04:42,063 - INFO - Using device: mps
2025-03-29 17:04:42,063 - INFO - Training standard conditional GAN...
Epoch 0: D_loss=1.3601, G_loss=0.6270                                                                                                                    
Epoch 200: D_loss=1.2461, G_loss=0.7067                                                                                                                  
Epoch 400: D_loss=1.0840, G_loss=0.8104                                                                                                                  
Epoch 600: D_loss=0.9104, G_loss=0.9090                                                                                                                  
Epoch 800: D_loss=0.9407, G_loss=1.0468                                                                                                                  
Epoch 1000: D_loss=0.7606, G_loss=1.3781                                                                                                                 
Epoch 1200: D_loss=1.4139, G_loss=0.9502                                                                                                                 
Epoch 1400: D_loss=0.9325, G_loss=1.4357                                                                                                                 
Epoch 1600: D_loss=1.0888, G_loss=1.5970                                                                                                                 
Epoch 1800: D_loss=0.8257, G_loss=1.3777                                                                                                                 
Epoch 2000: D_loss=1.0171, G_loss=0.6711                                                                                                                 
Epoch 2200: D_loss=0.9521, G_loss=1.2012                                                                                                                 
Epoch 2400: D_loss=0.9033, G_loss=1.4654                                                                                                                 
Epoch 2600: D_loss=0.6475, G_loss=1.5147                                                                                                                 
Epoch 2800: D_loss=0.9046, G_loss=1.0014                                                                                                                 
Epoch 3000: D_loss=1.2785, G_loss=0.9854                                                                                                                 
Epoch 3200: D_loss=1.0698, G_loss=1.1234                                                                                                                 
Epoch 3400: D_loss=1.3119, G_loss=0.8867                                                                                                                 
Epoch 3600: D_loss=1.2369, G_loss=0.7726                                                                                                                 
Epoch 3800: D_loss=1.4383, G_loss=0.7563                                                                                                                 
Epoch 4000: D_loss=1.3419, G_loss=0.7464                                                                                                                 
Epoch 4200: D_loss=1.3569, G_loss=0.7502                                                                                                                 
Epoch 4400: D_loss=1.3288, G_loss=0.7427                                                                                                                 
Epoch 4600: D_loss=1.3195, G_loss=0.7526                                                                                                                 
Epoch 4800: D_loss=1.3944, G_loss=0.7474                                                                                                                 
Epoch 5000: D_loss=1.3655, G_loss=0.7085                                                                                                                 
Epoch 5200: D_loss=1.4096, G_loss=0.7014                                                                                                                 
Epoch 5400: D_loss=1.3838, G_loss=0.6809                                                                                                                 
Epoch 5600: D_loss=1.3902, G_loss=0.7014                                                                                                                 
Epoch 5800: D_loss=1.3668, G_loss=0.7003                                                                                                                 
Epoch 6000: D_loss=1.3803, G_loss=0.7071                                                                                                                 
Epoch 6200: D_loss=1.4057, G_loss=0.7175                                                                                                                 
Epoch 6400: D_loss=1.3848, G_loss=0.7042                                                                                                                 
Epoch 6600: D_loss=1.4338, G_loss=0.6782                                                                                                                 
Epoch 6800: D_loss=1.3809, G_loss=0.6710                                                                                                                 
Epoch 7000: D_loss=1.4508, G_loss=0.6840                                                                                                                 
Epoch 7200: D_loss=1.4007, G_loss=0.6721                                                                                                                 
Epoch 7400: D_loss=1.4298, G_loss=0.6922                                                                                                                 
Epoch 7600: D_loss=1.4187, G_loss=0.7012                                                                                                                 
Epoch 7800: D_loss=1.4129, G_loss=0.7074                                                                                                                 
Epoch 8000: D_loss=1.3752, G_loss=0.7101                                                                                                                 
Epoch 8200: D_loss=1.4024, G_loss=0.6901                                                                                                                 
Epoch 8400: D_loss=1.3945, G_loss=0.7272                                                                                                                 
Epoch 8600: D_loss=1.3730, G_loss=0.7131                                                                                                                 
Epoch 8800: D_loss=1.3588, G_loss=0.7229                                                                                                                 
Epoch 9000: D_loss=1.3961, G_loss=0.7128                                                                                                                 
Epoch 9200: D_loss=1.3853, G_loss=0.7136                                                                                                                 
Epoch 9400: D_loss=1.4123, G_loss=0.6985                                                                                                                 
Epoch 9600: D_loss=1.4051, G_loss=0.6774                                                                                                                 
Epoch 9800: D_loss=1.4038, G_loss=0.6814                                                                                                                 
Training Standard GAN: 100%|██████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [01:25<00:00, 117.19it/s]
2025-03-29 17:06:08,156 - INFO - Training conformalized conditional GAN (C-GAN)...
Training Conformal GAN (cGAN): 100%|███████████████████████████████████████████████████████████████████████████████| 10000/10000 [03:03<00:00, 54.62it/s]
2025-03-29 17:09:11,278 - INFO - 
Evaluation Metrics Comparison:
2025-03-29 17:09:11,288 - INFO - 
             Metric      GAN    C-GAN
            KS_mean 0.138333 0.141667
   Wasserstein_mean 0.147386 0.162804
Downstream_Accuracy 0.966667 0.973333
```

### Evaluation of GAN vs C-GAN Performance

Based on the evaluation metrics, the **C-GAN** appears to be slightly better for generating high-quality synthetic data. Below is the reasoning behind this conclusion:

#### 1. **Downstream Accuracy**:
- **C-GAN** achieves a higher downstream accuracy (**97.33%**) compared to **GAN** (**96.67%**). This is the most important metric because it reflects how well a classifier trained on the synthetic data performs on real data. A higher downstream accuracy suggests that **C-GAN's synthetic data better captures the underlying patterns** in the real data, which is crucial for many applications that depend on high-quality synthetic data.

#### 2. **KS Mean**:
- The **Kolmogorov-Smirnov (KS)** statistic measures the maximum difference between the cumulative distribution functions of two distributions. **C-GAN** has a slightly higher KS mean (**0.141667**) compared to **GAN** (**0.138333**). 
- While the difference is minimal, **lower values** of KS indicate better similarity between the distributions. Since the values are close, this difference is negligible and doesn't significantly affect the conclusion.

#### 3. **Wasserstein Mean**:
- The **Wasserstein distance** (Earth Mover's Distance) measures how much "work" is needed to transform one distribution into another. **C-GAN** has a higher value (**0.162804**) compared to **GAN** (**0.147386**).
- Typically, **lower Wasserstein distances** indicate that the generated and real distributions are closer. While **C-GAN** has a higher Wasserstein mean, suggesting a slight disadvantage in distribution matching, this is not a deal-breaker given the other improvements.

### Overall Recommendation:
**C-GAN** should be preferred because:
- It provides better downstream task performance, which is often the ultimate goal in synthetic data generation.
- The improvement in downstream accuracy (+0.67%) outweighs the slight disadvantage in Wasserstein distance.
- The **conformalized approach** in **C-GAN** likely provides **additional theoretical guarantees** about the quality of the generated data that standard **GANs** do not offer.

While **C-GAN** has a slight disadvantage in the Wasserstein distance, its improved downstream accuracy suggests it is generating more useful synthetic data overall, making it the better choice.
