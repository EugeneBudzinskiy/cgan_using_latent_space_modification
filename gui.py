import io
import random
import time

import torch
import numpy as np
import streamlit as st
from PIL import Image

from generator.files import legacy
from encoder.files.train import Encoder

import torch_directml
DEVICE = torch_directml.device()
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


GENERATOR_MODEL_PATH = "generator/files/training-runs/network-snapshot.pkl"
ENCODER_MODEL_PATH = "encoder/files/checkpoint/model_at_0100.pth"

# Відповідності художнього стилю: (Назва для інтерфейсу, технічна назва, індекс класу)
ART_STYLES = [
    ("Art Nouveau", "art_nouveau", 0),
    ("Baroque", "baroque", 1),
    ("Expressionism", "expressionism", 2),
    ("Impressionism", "impressionism", 3),
    ("Post-Impressionism", "post_impressionism", 4),
    ("Realism", "realism", 5),
    ("Renaissance", "renaissance", 6),
    ("Romanticism", "romanticism", 7),
    ("Surrealism", "surrealism", 8),
    ("Ukiyo-e", "ukiyo_e", 9),
]


class ImageGenerator:
    """
    Клас, який інкапсулює логіку генерації зображень заданого стилю
    Використовує попередньо натреновану генеративну модель та енкодер класового зсуву
    """
    def __init__(self, device: torch.device | str):
        self.device = device
        self.z_dim = 256
        self.num_classes = 10

        self._generator = self.load_generator(network_pkl=GENERATOR_MODEL_PATH, device=self.device)
        self._encoder = self.load_encoder(
            model_path=ENCODER_MODEL_PATH, z_dim=self.z_dim, num_classes=self.num_classes, device=self.device
        )

    @staticmethod
    def load_generator(network_pkl: str, device: torch.device | str) -> torch.nn.Module:
        """
        Завантажує попередньо натреновану генеративну модель з pickle-файлу для подальшої генерації зображень
        :param network_pkl: str
           Шлях до pkl-файлу згенерованої мережі
        :param device: torch.device | str
           Обчислювальний пристрій, на який слід перенести генератор
        :return: torch.nn.Module
           Об'єкт генератора у режимі eval()
        """
        with open(network_pkl, mode="rb") as f:
            generator = legacy.load_network_pkl(f)["G_ema"].to(device)
        return generator.eval()

    @staticmethod
    def load_encoder(model_path: str, z_dim: int, num_classes: int, device: torch.device | str) -> torch.nn.Module:
        """
        Завантажує попередньо натренований енкодер з checkpoint-файлу
        :param model_path: str
            Шлях до checkpoint-файлу
        :param z_dim: int
            Розмірність латентного простору
        :param num_classes: int
            Кількість класів
        :param device: torch.device | str
            Обчислювальний пристрій
        :return: torch.nn.Module
            Енкодер у режимі eval
        """
        encoder = Encoder(num_classes=num_classes, z_dim=z_dim)
        encoder.to(device)

        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        encoder.load_state_dict(state_dict=state_dict, strict=True)
        return encoder.eval()

    def generate_image(self, latent_vector: torch.Tensor,
                       truncation_psi: float = 1.0, noise_mode: str = "const") -> np.ndarray:
        """
        Генерує одне зображення з латентним вектором за допомогою генератора
        :param latent_vector: torch.Tensor
            Латентний вектор для генерації зображення
        :param truncation_psi: float
            Параметр тримування генератора
        :param noise_mode: str
            Параметр конкретизації режиму шуму
        :return: np.ndarray
            Масив зображення у форматі HWC, uint8
        """
        label = torch.zeros([1, self._generator.c_dim], device=self.device)
        with torch.no_grad():
            img = self._generator(latent_vector, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            return img[0].cpu().numpy()

    def generate_images_with_style(self, style_idx: int, seed: int, n: int) -> list[Image]:
        """
        Генерує список зі n зображень, кожне із зазначеним стилем (через зсув енкодера класу)
        :param style_idx: int
            Індекс класу стилю
        :param seed: int
            Значення зерна (або None - для випадкового)
        :param n: int
            Кількість згенерованих зображень
        :return: list[Image]
            Список об'єктів PIL.Image
        """
        images = []
        rng = torch.random.manual_seed(seed if seed is not None else random.randint(0, 2_000_000_000))
        for i in range(n):
            z = torch.randn(1, self.z_dim, generator=rng, device=self.device)

            style_tensor = torch.tensor([style_idx], device=self.device)
            delta = self._encoder(style_tensor)
            new_z = z + delta

            img = self.generate_image(latent_vector=new_z)
            img = Image.fromarray(img)
            images.append(img)
        return images


def main():
    if st.session_state.get("image_generator") is None:
        st.session_state["image_generator"] = ImageGenerator(device=DEVICE)
    image_generator = st.session_state["image_generator"]

    st.set_page_config(page_title="Генератор зображень", layout="wide")

    st.title("🎨 Генератор умовних зображень")
    st.markdown("Виберіть художній стиль, задайте параметри та згенеруйте серію унікальних картин")

    # --- Бічна панель з налаштуваннями ---
    with st.sidebar:
        st.header("Налаштування")
        style_display = [n for n, _, __ in ART_STYLES]
        idx = st.selectbox("Стиль генерації:", range(len(ART_STYLES)), format_func=lambda x: style_display[x])

        use_fixed_seed = st.checkbox("Фіксоване зерно генерації", value=False)
        seed = st.number_input("Значення зерна", min_value=0, max_value=2_000_000_000, value=1234567, disabled=not use_fixed_seed)

        num_images = st.slider("Кількість згенерованих зображень в серії", 1, 20, 3)

        st.markdown("---")
        generate_btn = st.button("Згенерувати")

    # --- Стартова ініціалізація ---
    if "images" not in st.session_state:
        st.session_state["images"] = []
    if "last_params" not in st.session_state:
        st.session_state["last_params"] = None
    if "selected_flags" not in st.session_state:
        st.session_state["selected_flags"] = []

    # --- Визначення параметрів для генерації ---
    params = (idx, seed, num_images)
    if generate_btn:
        style_class = ART_STYLES[idx][2]
        seed_value = int(seed) if use_fixed_seed else None

        # Згенерувати серію зображень
        start_time = time.perf_counter()
        images = image_generator.generate_images_with_style(style_idx=style_class, seed=seed_value, n=num_images)
        total_time = time.perf_counter() - start_time
        per_image_time = total_time / num_images if num_images else 0

        st.session_state["images"] = images
        st.session_state["generation_time"] = (total_time, per_image_time)
        st.session_state["last_params"] = params

        # Скидання вибраних (selected_flags) після генерації
        st.session_state["selected_flags"] = [False] * num_images

    # --- Галерея зображень й опція збереження ---
    if st.session_state["images"]:
        st.subheader("Згенеровані зображення:")

        imgs = st.session_state["images"]
        selected_flags = st.session_state.get("selected_flags", [False] * len(imgs))

        # --- Кнопки "Вибрати всі" та "Скинути вибір" ---
        col_select_all, col_reset_all = st.columns([1, 1])
        with col_select_all:
            if st.button("✅ Вибрати всі", use_container_width=True):
                st.session_state["selected_flags"] = [True] * len(imgs)
                selected_flags = st.session_state["selected_flags"]
        with col_reset_all:
            if st.button("🗙 Скинути вибір", use_container_width=True):
                st.session_state["selected_flags"] = [False] * len(imgs)
                selected_flags = st.session_state["selected_flags"]

        # --- Галерея ---
        per_row = 5
        cols = st.columns(min(len(imgs), per_row))
        new_selected_flags = list(selected_flags)

        for i, img in enumerate(imgs):
            with cols[i % len(cols)]:
                st.image(img, use_container_width=True)
                new_selected_flags[i] = st.checkbox(f"Зберегти #{i + 1}", key=f"save_{i}", value=selected_flags[i])

        # Оновлюємо сесію
        st.session_state["selected_flags"] = new_selected_flags
        selected = [img for img, checked in zip(imgs, new_selected_flags) if checked]

        # --- Завантаження обраних ---
        if selected:
            buf = io.BytesIO()
            if len(selected) == 1:
                selected[0].save(buf, format="PNG")
                download_code = st.download_button(
                    label="⬇️ Завантажити",
                    data=buf.getvalue(),
                    file_name="image.png",
                    mime="image/png"
                )
            else:
                import zipfile
                with zipfile.ZipFile(buf, "w") as zf:
                    for j, img in enumerate(selected):
                        img_bytes = io.BytesIO()
                        img.save(img_bytes, format="PNG")
                        zf.writestr(f"image_{j + 1}.png", img_bytes.getvalue())
                download_code = st.download_button("⬇️ Завантажити ZIP", data=buf.getvalue(), file_name="images.zip",
                                                   mime="application/zip")

            # --- Скидання вибраних після завантаження ---
            if download_code:
                st.session_state["selected_flags"] = [False] * len(imgs)

        # --- Показати час генерації ---
        if "generation_time" in st.session_state:
            total_time, per_image_time = st.session_state["generation_time"]
            st.info(f"⏱️ Час генерації: {total_time:.2f} сек (≈ {per_image_time:.2f} сек/зображення)")


if __name__ == "__main__":
    main()
