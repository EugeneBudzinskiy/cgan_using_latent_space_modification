# Conditional GAN Image Generator with Latent Space Modification

This repository provides a pipeline for conditional image generation in various art styles, featuring custom-trained
StyleGAN2, encoder (style embedding), mapper (residual blocks), classifier (ResNet-based), and a Streamlit-based GUI.  
**Part of model code in `generator/files` is based on [ProjectedGAN](https://github.com/autonomousvision/projected-gan). 
The main dataset is [ArtBench-10](https://github.com/liaopeiyuan/artbench).**

## Features

- Conditional image generation for 10 art styles ([ArtBench-10](https://github.com/liaopeiyuan/artbench))
- Custom-trained encoder to control latent space (style manipulation)
- ResNet-based classifier for style recognition in the latent/image space
- Residual block mapper for latent vector -> style mapping
- Modular training scripts
- Streamlit GUI for interactive style-based generation and batch image downloads

---

## Folder Structure

```
classificator/                # Classifier training and checkpoints
create_latent_dataset/        # Scripts for latent dataset creation/processing
dataset/                      # Contains dataset zips and labels (ArtBench-10)
encoder/                      # Encoder training and checkpoints
generator/files/              # Fork (copy) of ProjectedGAN codebase
mapper/                       # Mapper network and checkpoints
gui.py                        # Streamlit web UI for image generation
...
```
- **`generator/files`** is a forked copy of [ProjectedGAN](https://github.com/autonomousvision/projected-gan), with your trained weights in `training-runs/`.
- **`dataset/artbench256-60k-split`** and `dataset/artbench256_10k.zip`: [ArtBench-10](https://github.com/liaopeiyuan/artbench) data.

---

## Setup

### 1. Clone Repository

```bash
git clone <this-repo-url>
cd <repo-folder>
```

### 2. (Optional) Download Datasets

ArtBench-10:
- Download [ArtBench-10](https://github.com/liaopeiyuan/artbench) (full and 10k subset) as described in their repo.
- Place dataset files in the `dataset/artbench256-60k-split` and `dataset/artbench256_10k.zip` respectfully.

Custom Latent Dataset
- Download [latent_dataset](https://drive.google.com/file/d/1gi4HFAg3RuQvCwAyaRZ058kCzvh4vOeJ/view?usp=sharing) zip
- Place dataset files in the `dataset/latent_dataset` structure as shown above by folder tree.

> **Note:** For full reproducibility of training pipelines, these datasets are required. For GUI/demo image generation, pre-trained weights are sufficient.

### 3. Install Dependencies

**Requires:** Python 3.11, Windows OS recommended, GPU (AMD with DirectML or CUDA).
```bash
pip install -r requirements.txt
```
> The code can use AMD GPU via [torch_directml](https://github.com/microsoft/DirectML).  
> If you have NVIDIA GPU and want to use CUDA, you can comment `torch_directml` and uncomment the `torch.device("cuda:0"...)` lines in code.

---

### 4. Download Pretrained Model Weights

#### **Generator (StyleGAN2, ProjectedGAN fork)**

- Download [generator checkpoint]([https://drive.google.com/file/d/19_8eLCIBUCpmhO9PWV39bzkdj5Xz7gEp/view?usp=sharing])  
  Place at: `generator/files/training-runs/network-snapshot.pkl`

#### **Encoder**
- Download [encoder checkpoint]([https://drive.google.com/file/d/10UwNxmLVGkv39akk-4F4vL5UGZo9DpNo/view?usp=sharing])  
  Place at: `encoder/files/checkpoint/model_at_0100.pth`

#### **Classifier**
- Download [classifier checkpoint]([https://drive.google.com/file/d/1j7lacpg6UAIB1IlRV9JMdSN90FIKwVPH/view?usp=sharing])  
  Place at: `classificator/files/checkpoint/model_at_0013.pth`

#### **Mapper**
- Download [mapper checkpoint]([https://drive.google.com/file/d/1RUJI3VIAYck12UjG4pMovtlN1MQBkRVw/view?usp=sharing])  
  Place at: `mapper/files/checkpoint/model_at_0217.pth`

---

## Launching the GUI

```bash
streamlit run gui.py
```
- The GUI will open in your browser.
- Select the art style, number of images, seed, and click **Generate**.
- You can download selected images as PNG or ZIP archive.

---

## Notes and Tips

- The project was developed and partially trained on Windows with AMD GPU (DirectML).  
  However, CUDA GPUs **can** also be used with minimal code changes.
- All code should run in the prepared `.venv` environment.
- For full experiments or training, make sure you have all necessary datasets.
- Cite original works when publishing or reusing the code:
  - **[ProjectedGAN](https://github.com/autonomousvision/projected-gan)**
  - **[ArtBench-10 dataset](https://github.com/liaopeiyuan/artbench)**

## License and Attribution

- The code in `generator/files` is from [ProjectedGAN](https://github.com/autonomousvision/projected-gan), licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
- The remaining custom code and scripts in this repository are released under the MIT License.  
  **However, because of ProjectedGAN's non-commercial clause, this repository as a whole **may NOT be used for commercial purposes**.
- The [ArtBench-10 dataset](https://github.com/liaopeiyuan/artbench) is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
---

*For any questions or to report issues, please use Issues section or contact the maintainer.*