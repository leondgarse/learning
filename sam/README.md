- [Github facebookresearch/segment-anything(https://github.com/facebookresearch/segment-anything)
- [Github ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
```py
!wget https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt

import torch
from torch.nn import functional as F
from mask_decoder import MaskDecoder
from prompt_encoder import PromptEncoder
from transformer import TwoWayTransformer
from tiny_vit_sam import TinyViT
from copy import deepcopy
from PIL import Image
from skimage.data import astronaut

device = "cuda" if torch.cuda.is_available() else "cpu"
image_encoder = TinyViT(img_size=1024)
prompt_encoder = PromptEncoder()
mask_decoder = MaskDecoder(transformer=TwoWayTransformer())

ss = torch.load("mobile_sam.pt")
image_encoder.load_state_dict({kk[len('image_encoder.'):]: vv for kk, vv in ss.items() if kk.startswith('image_encoder.')})
prompt_encoder.load_state_dict({kk[len('prompt_encoder.'):]: vv for kk, vv in ss.items() if kk.startswith('prompt_encoder.')})
mask_decoder.load_state_dict({kk[len('mask_decoder.'):]: vv for kk, vv in ss.items() if kk.startswith('mask_decoder.')})

def preprocess(inputs, image_encoder_size=1024) -> torch.Tensor:
    """Normalize pixel values and pad to a square input."""
    # Normalize colors
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(-1, 1, 1)
    inputs = (inputs - pixel_mean) / pixel_std

    # Pad
    h, w = inputs.shape[-2:]
    padh = image_encoder_size - h
    padw = image_encoder_size - w
    inputs = F.pad(inputs, (0, padw, 0, padh))
    return inputs

def postprocess_masks(masks, image_encoder_size, input_size, original_size):
    """ Remove padding and upscale masks to the original image size. """
    masks = F.interpolate(masks, (image_encoder_size, image_encoder_size), mode="bilinear", align_corners=False)
    masks = masks[..., : input_size[0], : input_size[1]]
    masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
    return masks

def get_preprocess_shape(oldh, oldw, long_side_length):
    """ Compute the output size given input size and target long side length. """
    scale = long_side_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    return (newh, neww)

def apply_image(image, target_length=1024):
    """ Expects a numpy array with shape HxWxC in uint8 format. """
    target_size = get_preprocess_shape(image.shape[0], image.shape[1], target_length)
    return np.array(Image.fromarray(image).resize(target_size[::-1]))

def apply_coords(coords, original_size, target_length=1024) -> np.ndarray:
    """ Expects a numpy array of length 2 in the final dimension. Requires the original image size in (H, W) format. """
    old_h, old_w = original_size
    new_h, new_w = get_preprocess_shape(original_size[0], original_size[1], target_length)
    coords = deepcopy(coords).astype(float)
    coords[..., 0] = coords[..., 0] * (new_w / old_w)
    coords[..., 1] = coords[..., 1] * (new_h / old_h)
    return coords

def apply_boxes(boxes, original_size):
    """ Expects a numpy array shape Bx4. Requires the original image size in (H, W) format. """
    boxes = apply_coords(boxes.reshape(-1, 2, 2), original_size)
    return boxes.reshape(-1, 4)

def set_image(image, image_encoder_size=1024, image_format="RGB"):
    if image_format != "RGB":
        image = image[..., ::-1]

    # Transform the image to the form expected by the model
    original_size = image.shape[:2]
    input_image = apply_image(image)
    input_image_torch = torch.as_tensor(input_image, device=device)
    transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

    # set_torch_image
    input_size = tuple(transformed_image.shape[-2:])
    input_image = preprocess(transformed_image, image_encoder_size=image_encoder_size)
    features = image_encoder(input_image)
    return features, original_size, input_size

def predict(features, original_size, input_size, image_encoder_size=1024, multimask_output=True, return_logits=False, mask_threshold=0.0):
    points, labels, boxes, mask_inputs = np.array([[400, 400]]), np.array([1]), None, None

    points = None if points is None else torch.as_tensor(apply_coords(points, original_size), dtype=torch.float, device=device)[None, :]
    labels = None if labels is None else torch.as_tensor(labels, dtype=torch.int, device=device)[None, :]
    boxes = None if boxes is None else torch.as_tensor(apply_boxes(boxes, original_size), dtype=torch.float, device=device)[None, :]
    mask_inputs = None if mask_inputs is None else torch.as_tensor(mask_inputs, dtype=torch.float, device=device)[None, :]

    # Embed prompts
    points = None if points is None else (points, labels)
    sparse_embeddings, dense_embeddings = prompt_encoder(points=points, boxes=boxes, masks=mask_inputs)

    # Predict masks
    low_res_masks, iou_predictions = mask_decoder(
        image_embeddings=features,
        image_pe=prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=multimask_output,
    )

    # Upscale the masks to the original image resolution
    masks = postprocess_masks(low_res_masks, image_encoder_size, input_size, original_size)

    if not return_logits:
        masks = masks > mask_threshold

    masks_np = masks[0].detach().cpu().numpy()
    iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
    low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
    return masks_np, iou_predictions_np, low_res_masks_np

image = astronaut()
point_coords = np.array([[400, 400]])
point_labels = np.array([1])
features, original_size, input_size = set_image(image)
masks, iou_predictions, low_res_masks = predict(features, original_size, input_size)
print(features.shape, original_size, input_size, masks.shape, iou_predictions.shape, low_res_masks.shape)
# torch.Size([1, 256, 64, 64]) (1367, 2048) (684, 1024) (3, 1367, 2048) (3,) (3, 256, 256)
```
```py
def show_mask(mask, ax, random_color=False):
    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else np.array([30/255, 144/255, 255/255, 0.6])
    mask_image = np.expand_dims(mask, -1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points, neg_points = coords[labels==1], coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    

for id, (mask, iou_prediction) in enumerate(zip(masks, iou_predictions)):
    fig = plt.figure(figsize=(10,10))
    plt.imshow(image)
    show_mask(mask, plt.gca())
    show_points(point_coords, point_labels, plt.gca())
    plt.title(f"Mask {id+1}, Score: {iou_prediction:.3f}", fontsize=18)
    plt.axis('off')
    plt.show()
fig.savefig('aa.jpg')
```
