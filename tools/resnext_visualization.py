import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import mmcv
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    # top_k_accuracy_score,
    confusion_matrix,
    classification_report
)

from mmpretrain.apis import ImageClassificationInferencer
from mmpretrain import init_model
from captum.attr import IntegratedGradients

# =========================
# 1. Configurations
# =========================

CONFIG = r'configs/resnext/my_resnext50_folder.py'
CHECKPOINT = r'work_dirs/resnext50_mydata/epoch_40.pth'
TEST_ROOT = r'D:\ECE 549 Final\Archive\test'

OUT_DIR = 'visualization/outputs'
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# =========================
# 2. Load model & inferencer
# =========================

print('Loading model...')
model = init_model(CONFIG, CHECKPOINT, device=DEVICE)
model.eval()

inferencer = ImageClassificationInferencer(
    model=model,
    device=DEVICE,
    progress=False
)

# =========================
# 3. Collect test predictions
# =========================

print('Running inference on test set...')

y_true, y_pred, y_score = [], [], []
img_paths = []

class_names = sorted(os.listdir(TEST_ROOT))
all_images = []

for cls_idx, cls_name in enumerate(class_names):
    cls_dir = os.path.join(TEST_ROOT, cls_name)
    if not os.path.isdir(cls_dir):
        continue
    for img_name in os.listdir(cls_dir):
        if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            all_images.append((cls_idx, os.path.join(cls_dir, img_name)))

'''
# Optional: Subsample test set for faster visualization
import random

TEST_RATIO = 0.025  
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
random.shuffle(all_images)

num_test = int(len(all_images) * TEST_RATIO)
all_images = all_images[:num_test]
print(f'Using {num_test} / {len(class_names)} test images '
      f'({TEST_RATIO * 100:.0f}% of test set)')
'''

for cls_idx, img_path in tqdm(all_images, desc='Inference on test set'):
    result = inferencer(img_path)[0]

    y_true.append(cls_idx)
    y_pred.append(result['pred_label'])

    with torch.no_grad():
        data = list(inferencer.preprocess([img_path]))[0]
        logits = model.test_step(data)[0].pred_score.cpu().numpy()

    y_score.append(logits)
    img_paths.append(img_path)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.stack(y_score)

# =========================
# Fixed samples for visualization
# =========================

correct_indices = [
    i for i in range(len(y_true))
    if y_true[i] == y_pred[i]
]

VIS_INDICES = correct_indices[:3]
print('Visualization sample indices:', VIS_INDICES)

# =========================
# 4. Metrics
# =========================

top1 = accuracy_score(y_true, y_pred)
top5 = np.mean([
    y_true[i] in np.argsort(y_score[i])[-5:]
    for i in range(len(y_true))
])

report = classification_report(
    y_true,
    y_pred,
    digits=4
)

with open(os.path.join(OUT_DIR, 'metrics.txt'), 'w') as f:
    f.write(f'Top-1 Accuracy: {top1:.4f}\n')
    f.write(f'Top-5 Accuracy: {top5:.4f}\n\n')
    f.write(report)

print('Metrics saved.')

# =========================
# 5. Confusion Matrix
# =========================

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, cmap='Blues', xticklabels=False, yticklabels=False)
plt.title('Confusion Matrix (Stanford Dogs)')
plt.xlabel('Predicted')
plt.ylabel('Ground Truth')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'))
plt.close()

print('Confusion matrix saved.')

# =========================
# 6. Grad-CAM for ResNeXt (CNN) with data_preprocessor
# =========================

print('Running Grad-CAM for ResNeXt (data_preprocessor aligned)...')

import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

cam_dir = os.path.join(OUT_DIR, 'gradcam')
os.makedirs(cam_dir, exist_ok=True)

def find_last_conv_layer(module: torch.nn.Module) -> torch.nn.Module:
    """Robustly find the last Conv2d layer inside a module."""
    last_conv = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError('No Conv2d layer found. Cannot run Grad-CAM.')
    return last_conv

# 推荐：对 ResNeXt 选 backbone 最后一层卷积
# 如果你 backbone 结构不同，这个函数也能兜底找到最后一个 Conv2d
target_layer = find_last_conv_layer(model.backbone)
cam = GradCAM(model=model, target_layers=[target_layer])

# 用于拼图（3 张并排）
panel_imgs = []

for vis_id, i in enumerate(VIS_INDICES):
    # 读原图（RGB）
    img_bgr = mmcv.imread(img_paths[i])       # BGR
    img_rgb = mmcv.bgr2rgb(img_bgr)           # RGB
    H, W = img_rgb.shape[:2]

    # resize 到 224（和你的 pipeline 一致）
    img_resized = mmcv.imresize(img_rgb, (224, 224))
    rgb_float = img_resized.astype(np.float32) / 255.0

    # 转 tensor（注意：这里只是 raw tensor，归一化交给 data_preprocessor）
    img_tensor = (
        torch.tensor(img_resized)
        .permute(2, 0, 1)
        .float()
        .to(DEVICE)
    )

    # 走 mmpretrain 的 data_preprocessor（保证 mean/std/to_rgb 与训练一致）
    with torch.no_grad():
        data = {'inputs': [img_tensor], 'data_samples': None}
        data = model.data_preprocessor(data, training=False)
        input_tensor = data['inputs']  # shape: (1, 3, 224, 224)

    target_class = int(y_pred[i])

    # Grad-CAM
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_class)]
    )[0]  # (224, 224), 0~1

    # overlay（在 224x224 上）
    overlay_224 = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    # 保存单张
    out_path = os.path.join(cam_dir, f'gradcam_{vis_id}.png')
    cv2.imwrite(out_path, cv2.cvtColor(overlay_224, cv2.COLOR_RGB2BGR))

    panel_imgs.append(overlay_224)

# 拼成 1x3 大图（像你截图那样）
fig, axes = plt.subplots(1, len(panel_imgs), figsize=(12, 4))
if len(panel_imgs) == 1:
    axes = [axes]

for ax, im in zip(axes, panel_imgs):
    ax.imshow(im)
    ax.axis('off')

plt.tight_layout()
panel_path = os.path.join(cam_dir, 'gradcam_panel.png')
plt.savefig(panel_path, dpi=300, bbox_inches='tight')
plt.close()

print(f'Grad-CAM saved to: {cam_dir}')
print(f'Panel saved to: {panel_path}')


# =========================
# 7. Integrated Gradients
# =========================

print('Running Integrated Gradients...')

ig_root = os.path.join(OUT_DIR, 'integrated_gradients')
ig_raw_dir = os.path.join(ig_root, 'raw')
ig_overlay_dir = os.path.join(ig_root, 'overlay')
ig_image_dir = os.path.join(ig_root, 'image')

os.makedirs(ig_raw_dir, exist_ok=True)
os.makedirs(ig_overlay_dir, exist_ok=True)
os.makedirs(ig_image_dir, exist_ok=True)

ig = IntegratedGradients(model)

for vis_id, i in enumerate(VIS_INDICES):
    img = mmcv.imread(img_paths[i])
    img_float = img.astype(np.float32) / 255.0
    H, W = img.shape[:2]

    img_resized = mmcv.imresize(img, (224, 224))

    img_tensor = (
        torch.tensor(img_resized)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(DEVICE)
    )

    baseline = torch.zeros_like(img_tensor)

    attr = ig.attribute(
        img_tensor,
        baseline,
        target=int(y_pred[i])
    )

    attr = attr.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    attr = np.abs(attr).mean(axis=2)

    attr -= attr.min()
    attr /= (attr.max() + 1e-8)

    attr_full = mmcv.imresize(attr, (W, H))
    heatmap = plt.get_cmap('jet')(attr_full)[:, :, :3]
    overlay = 0.6 * img_float + 0.4 * heatmap
    overlay = np.clip(overlay, 0, 1)

    plt.imsave(os.path.join(ig_raw_dir, f'ig_raw_{vis_id}.png'), attr_full, cmap='hot')
    plt.imsave(os.path.join(ig_overlay_dir, f'ig_overlay_{vis_id}.png'), overlay)
    plt.imsave(os.path.join(ig_image_dir, f'ig_image_{vis_id}.png'), img_float)

print('Integrated Gradients visualizations saved.')

print('✅ All visualizations completed successfully!')
