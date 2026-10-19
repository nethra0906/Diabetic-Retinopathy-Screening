import os

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import models, transforms


GRADES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_DIR,
    "models",
    "best_joint_finetuned_attention_fusion_repacked.pth",
)


class JointAttentionFusionModel(nn.Module):
    """Matches JointAttentionFusionModel from notebook 12."""

    def __init__(
        self,
        resnet_features,
        efficientnet_features,
        efficientnet_pool,
    ):
        super().__init__()

        self.resnet_features = resnet_features
        self.efficientnet_features = efficientnet_features
        self.efficientnet_pool = efficientnet_pool

        self.resnet_projection = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

        self.efficientnet_projection = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
        )

        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 5),
        )

    def forward(self, x):
        resnet_features = self.resnet_features(x)
        resnet_features = torch.flatten(resnet_features, 1)
        resnet_features = self.resnet_projection(resnet_features)

        efficientnet_features = self.efficientnet_features(x)
        efficientnet_features = self.efficientnet_pool(efficientnet_features)
        efficientnet_features = torch.flatten(efficientnet_features, 1)
        efficientnet_features = self.efficientnet_projection(
            efficientnet_features
        )

        combined_features = torch.cat(
            [resnet_features, efficientnet_features],
            dim=1,
        )

        attention_weights = torch.softmax(
            self.attention(combined_features),
            dim=1,
        )

        fused_features = (
            attention_weights[:, 0:1] * resnet_features
            + attention_weights[:, 1:2] * efficientnet_features
        )

        logits = self.classifier(fused_features)
        return logits, attention_weights


@st.cache_resource
def load_model(checkpoint_path):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint file not found:\n{checkpoint_path}"
        )

    resnet = models.resnet18(weights=None)
    resnet_features = nn.Sequential(
        *list(resnet.children())[:-1]
    )

    efficientnet = models.efficientnet_b0(weights=None)

    model = JointAttentionFusionModel(
        resnet_features=resnet_features,
        efficientnet_features=efficientnet.features,
        efficientnet_pool=nn.AdaptiveAvgPool2d((1, 1)),
    )

    # This checkpoint includes NumPy data in its training metadata.
    # Use weights_only=False only because this is your trusted checkpoint.
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(DEVICE)
    model.eval()

    return model


PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


def make_gradcam(model, image_tensor, target_layer, class_index):
    captured = {}

    def save_activations(_module, _inputs, output):
        captured["activations"] = output

    def save_gradients(_module, _grad_input, grad_output):
        captured["gradients"] = grad_output[0]

    forward_hook = target_layer.register_forward_hook(save_activations)
    backward_hook = target_layer.register_full_backward_hook(save_gradients)

    try:
        model.zero_grad(set_to_none=True)

        logits, _ = model(image_tensor)
        score = logits[0, class_index]
        score.backward()

        activations = captured["activations"].detach()
        gradients = captured["gradients"].detach()

        channel_weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = F.relu(
            (channel_weights * activations).sum(dim=1, keepdim=True)
        )

        cam = F.interpolate(
            cam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0].detach().cpu().numpy()

        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    finally:
        forward_hook.remove()
        backward_hook.remove()


def make_overlay(image_rgb, cam, alpha=0.42):
    from matplotlib import colormaps

    heatmap = (
        colormaps["jet"](np.clip(cam, 0, 1))[:, :, :3] * 255
    ).astype(np.uint8)

    overlay = (
        image_rgb.astype(np.float32) * (1 - alpha)
        + heatmap.astype(np.float32) * alpha
    )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def image_quality_checks(image):
    """Basic image heuristics; not a trained or validated quality model."""

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    brightness = float(gray.mean() / 255)
    contrast = float(gray.std() / 255)
    sharpness = None

    notes = []

    if min(image.size) < 128:
        notes.append("Low image resolution")

    if brightness < 0.12:
        notes.append("Image may be too dark")
    elif brightness > 0.88:
        notes.append("Image may be overexposed")

    if contrast < 0.08:
        notes.append("Low contrast")

    try:
        import cv2

        sharpness = float(
            cv2.Laplacian(gray, cv2.CV_64F).var()
        )

        if sharpness < 15:
            notes.append("Possible blur")
    except ImportError:
        pass

    if not notes:
        notes.append("No obvious issue found by basic checks")

    return brightness, contrast, sharpness, notes


st.set_page_config(
    page_title="Diabetic Retinopathy Screening",
    page_icon="👁️",
    layout="wide",
)

st.title("Diabetic Retinopathy Screening")
st.caption(
    "Research prototype. This result is not a diagnosis and does not "
    "replace assessment by an eye-care professional."
)

default_checkpoint = os.getenv(
    "DR_CHECKPOINT",
    DEFAULT_CHECKPOINT,
)

checkpoint_path = st.sidebar.text_input(
    "Model checkpoint path",
    value=default_checkpoint,
)

st.sidebar.markdown(
    """
    **Grade labels**

    0 · No DR  
    1 · Mild  
    2 · Moderate  
    3 · Severe  
    4 · Proliferative DR
    """
)

uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=["jpg", "jpeg", "png", "tif", "tiff"],
)

if uploaded_file is None:
    st.info("Upload a fundus image to see the model output.")
    st.stop()

try:
    image = ImageOps.exif_transpose(
        Image.open(uploaded_file)
    ).convert("RGB")

    image_tensor = PREPROCESS(image).unsqueeze(0).to(DEVICE)
    model = load_model(checkpoint_path)

    with torch.no_grad():
        logits, attention = model(image_tensor)
        scores = torch.softmax(logits, dim=1)[0].cpu().numpy()

    predicted_grade = int(scores.argmax())
    branch_weights = attention[0].detach().cpu().numpy()

    brightness, contrast, sharpness, quality_notes = (
        image_quality_checks(image)
    )

    image_column, result_column = st.columns([1, 1.3])

    with image_column:
        st.image(
            image,
            caption=f"Uploaded image · {image.width} × {image.height}",
            use_container_width=True,
        )

        st.subheader("Basic image checks")
        st.write(" · ".join(quality_notes))

        sharpness_text = (
            f" · Sharpness: {sharpness:.1f}"
            if sharpness is not None
            else ""
        )

        st.caption(
            f"Brightness: {brightness:.2f}"
            f" · Contrast: {contrast:.2f}"
            f"{sharpness_text}"
        )

        st.caption(
            "These are simple image heuristics, not a validated "
            "image-quality model."
        )

    with result_column:
        st.metric(
            "Predicted DR grade",
            f"{predicted_grade} · {GRADES[predicted_grade]}",
            f"{scores[predicted_grade]:.1%} model score",
        )

        # Demo screening flag only. Confirm any clinical threshold
        # with qualified clinicians before real-world use.
        if predicted_grade >= 2:
            st.error(
                "Grade 2 or higher screening flag. "
                "Arrange professional assessment."
            )
        else:
            st.info(
                "The model predicted below grade 2. "
                "This does not rule out diabetic retinopathy."
            )

        st.subheader("Model scores by grade")
        st.bar_chart({
            GRADES[i]: float(scores[i])
            for i in range(len(GRADES))
        })

        st.subheader("Model branch attention")
        st.write({
            "ResNet-18": f"{branch_weights[0]:.1%}",
            "EfficientNet-B0": f"{branch_weights[1]:.1%}",
        })

    with st.expander("Grad-CAM explanations", expanded=True):
        st.caption(
            "Grad-CAM highlights image regions that influenced the selected "
            "class score. It is not a lesion mask or a diagnosis."
        )

        with st.spinner("Generating Grad-CAM maps..."):
            resnet_cam = make_gradcam(
                model,
                image_tensor,
                model.resnet_features[-2],  # ResNet layer4
                predicted_grade,
            )

            efficientnet_cam = make_gradcam(
                model,
                image_tensor,
                model.efficientnet_features[-1],
                predicted_grade,
            )

            fused_cam = (
                branch_weights[0] * resnet_cam
                + branch_weights[1] * efficientnet_cam
            )

            fused_min = float(fused_cam.min())
            fused_max = float(fused_cam.max())

            if fused_max - fused_min > 1e-8:
                fused_cam = (fused_cam - fused_min) / (
                    fused_max - fused_min
                )
            else:
                fused_cam = np.zeros_like(fused_cam)

        resized_image = image.resize((224, 224))
        base_image = np.asarray(resized_image)

        cam_column_1, cam_column_2, cam_column_3 = st.columns(3)

        cam_column_1.image(
            make_overlay(base_image, resnet_cam),
            caption="ResNet-18 Grad-CAM",
            use_container_width=True,
        )

        cam_column_2.image(
            make_overlay(base_image, efficientnet_cam),
            caption="EfficientNet-B0 Grad-CAM",
            use_container_width=True,
        )

        cam_column_3.image(
            make_overlay(base_image, fused_cam),
            caption="Attention-weighted Grad-CAM",
            use_container_width=True,
        )

    summary = [
        f"Predicted grade: {predicted_grade} ({GRADES[predicted_grade]})",
        f"Top-grade model score: {scores[predicted_grade]:.6f}",
        "Grade scores: "
        + ", ".join(
            f"{i}={score:.6f}" for i, score in enumerate(scores)
        ),
        (
            "Branch attention: "
            f"ResNet-18={branch_weights[0]:.6f}, "
            f"EfficientNet-B0={branch_weights[1]:.6f}"
        ),
        "Research prototype output; not a diagnosis.",
    ]

    st.download_button(
        "Download result summary",
        data="\n".join(summary),
        file_name="dr_screening_result.txt",
        mime="text/plain",
    )
except Exception as error:
    st.error(f"Could not run inference: {error}")
