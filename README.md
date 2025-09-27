## A Training Method for Improving Image Diversity in Generative Adversarial Networks Using a Perceptual Cosine Similarity Loss(LatentFlowGAN) <br> <sub>Official PyTorch implementation of Journal of the Korea Institute of Information and Communication
Engineering, vol. 29, issue 10, 2025 </sub>
<p align="center">
  <img width="380" height="400" alt="Image1" src="https://github.com/user-attachments/assets/5382e1c0-bf6d-4f85-8a03-a146083dd52a" />
  <img width="290" height="400" alt="Image2" src="https://github.com/user-attachments/assets/71fc1596-8deb-49e7-9ab9-5e8ee7cb2448" />
</p>

### <b>Abstract</b><br>
Generative Adversarial Networks (GANs) demonstrate strong performance in high-quality image synthesis but still suffer from mode collapse, where only a limited range of samples is generated. This study proposes a diversity loss based on cosine similarity that can be applied without modifying the network architecture. The proposed loss reduces redundant generation by minimizing the cosine similarity between feature vectors extracted from generated images, thereby enhancing diversity. This method allows for the preservation of semantic continuity in the latent space while encouraging the generation of more varied outputs. The effectiveness of the proposed loss is evaluated on the LSUN Bedroom and CelebA datasets using the DCGAN architecture. Experimental results based on FID and LPIPS metrics confirm that our method can effectively improve both the diversity and quality of generated images. Notably, these improvements are achieved without any changes to the network architecture, relying solely on the addition of a simple loss term during training.

<br>
Paper Link : [paper.pdf](https://github.com/user-attachments/files/22571391/paper.pdf)
