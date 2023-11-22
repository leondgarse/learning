import math
import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format, initializers
from keras_cv_attention_models.attention_layers import BiasLayer

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


""" Split """


class PositionEmbeddingRandom:
    def __init__(self, num_pos_feats=64, scale=-1):
        self.scale = scale if scale > 0 else 1
        self.positional_encoding_gaussian_matrix = self.scale * np.random.norm(size=[2, num_pos_feats]).astype("float32")

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size):
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(self, coords_input, image_size):
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

def PointsEncoder(embed_dim=256):
    rr = []
    for name in ["points_height", "points_width", "bboxes_height", "bboxes_width", "empty_points", "empty_bboxes"]:
        embeddings_initializer = initializers.RandomNormal(mean=0, stddev=1)
        bias_layer = attention_layers.BiasLayer(axis=-1, initializer=embeddings_initializer, name=name+"_bias")
        rr.append(models.Sequential([layers.Input([embed_dim]), bias_layer], name=name))
    return rr

def BoxesEncoder(embed_dim=256):
    return lambda xx: initializers.zeros()([xx.shape[0], embed_dim])

def MasksEncoder(embed_dim=256, mask_in_chans=16):
    inptus = layers.Input([None, None])
    conv2d_no_bias(1, mask_in_chans // 4, kernel_size=2, stride=2),
    mask_downscaling = models.Sequential([
        LayerNorm2d(mask_in_chans // 4),
        activation(),
        nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
        LayerNorm2d(mask_in_chans),
        activation(),
        nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
    ])


class SAM:
    def __init__(self, embed_dim=256, image_embedding_size=(64, 64), input_image_size=(1024, 1024), mask_in_chans=16):
        self.image_encoder = models.TinyViT_5M()
        # self.prompt_encoder =
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )

        scale, num_pos_feats = 1.0, 64
        scale = 1.0 if scale is None or scale <= 0.0 else scale
        self.positional_encoding_gaussian_matrix = scale * np.random.normal(size=[2, num_pos_feats])

    def _pe_encoding(self, coords):
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size):
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def normalize_coords(self, coords):
        coords = coords / [self.input_image_size[1], self.input_image_size[0]]
        coords = (2 * coords - 1) * (2 * np.pi)
        coords = coords @ self.positional_encoding_gaussian_matrix
        return np.concatenate([np.sin(coords), np.cos(coords)], axis=-1)

    def _embed_points(self, points, labels, pad=False):
        """Embeds point prompts."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            points = np.pad(points, [[0, 0], [0, 1], [0, 0]])
            labels = np.pad(labels, [[0, 0], [0, 1]], constant_values=-1)
        point_embedding = self.normalize_coords(points)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        return point_embedding

    def _embed_boxes(self, boxes):
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks):
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def __call__(self, image, points=None, labels=None, boxes=None, masks=None):
        batch_size, embed_dim = 1, 320
        points_inputs = self._embed_points(points, labels, pad=(boxes is None)) if points is not None else np.empty([batch_size, 0, 2])
        boxes_inputs = self._embed_boxes(boxes) if boxes is not None else np.empty([batch_size, 0, 2, 2])
        masks_inputs = self._embed_masks(masks) if masks is not None else np.empty([batch_size, 0, 256, 256, 1])
        # sparse_embeddings, dense_embeddings = functional.concat([points_inputs, boxes_inputs]), masks_inputs

        low_res_masks, iou_predictions = mask_decoder(
            image_embeddings=features,
            image_pe=prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )



        inputs = inputs.copy()

        # Get the batch shape based on the image input
        B = ops.shape(inputs["images"])[0]

        # The type of the placeholders must match the existing inputs with respect
        # to whether or not they are tensors (as opposed to Numpy arrays).
        zeros = ops.zeros if ops.is_tensor(inputs["images"]) else np.zeros

        # Fill in missing inputs.
        if "points" not in inputs:
            inputs["points"] = zeros((B, 0, 2))
        if "labels" not in inputs:
            inputs["labels"] = zeros((B, 0))
        if "boxes" not in inputs:
            inputs["boxes"] = zeros((B, 0, 2, 2))
        if "masks" not in inputs:
            inputs["masks"] = zeros((B, 0, 256, 256, 1))
