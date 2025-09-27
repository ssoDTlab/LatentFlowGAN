## A Training Method for Improving Image Diversity in Generative Adversarial Networks Using a Perceptual Cosine Similarity Loss(LatentFlowGAN) <br> <sub>Official PyTorch implementation of Journal of the Korea Institute of Information and Communication Engineering, vol. 29, issue 10, 2025 </sub>

These formulas describe the **latent space continuity**:

1. **Squared Distance**
$$\| z_1 - z_2 \|^2 = \sum_{i=1}^{100} ( z_{1,i} - z_{2,i} )^2$$
This is the Euclidean distance squared between two 100-dimensional latent vectors.

2. **Distribution**
$$\| z_1 - z_2 \|^2 \sim 2 \chi^2(100)$$
The squared distance follows twice a chi-square distribution with 100 degrees of freedom.

3. **Expected Distance**
$$\mathbb{E}\left[ \| z_1 - z_2 \| \right] \approx 14$$
The mean distance is about 14, showing that two random latent points are typically far apart, which supports the latent space’s smooth semantic continuity.

<p align="center">
  <img width="400" height="400" alt="Image" src="https://github.com/user-attachments/assets/db0a2f52-271d-438b-be4d-0933a9468cf8" />
</p>

### <b>Abstract</b><br>
Generative Adversarial Networks (GANs) demonstrate strong performance in high-quality image synthesis but still suffer from mode collapse, where only a limited range of samples is generated. This study proposes a diversity loss based on cosine similarity that can be applied without modifying the network architecture. The proposed loss reduces redundant generation by minimizing the cosine similarity between feature vectors extracted from generated images, thereby enhancing diversity. This method allows for the preservation of semantic continuity in the latent space while encouraging the generation of more varied outputs. The effectiveness of the proposed loss is evaluated on the LSUN Bedroom and CelebA datasets using the DCGAN architecture. Experimental results based on FID and LPIPS metrics confirm that our method can effectively improve both the diversity and quality of generated images. Notably, these improvements are achieved without any changes to the network architecture, relying solely on the addition of a simple loss term during training.

<br>
Paper : https://github.com/user-attachments/files/22571391/paper.pdf
