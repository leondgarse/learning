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

        self.points_height, self.points_width, self.bboxes_height, self.bboxes_width, self.empty_points, self.empty_bboxes = [ii.numpy() for ii in PointsEncoder()]

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
        coords = coords @ self.positional_encoding_gaussian_matrix  # [1, 1, 2] @ [2, 128] -> [1, 1, 128]
        return np.concatenate([np.sin(coords), np.cos(coords)], axis=-1)  # [1, 1, 256]

    def _embed_points_prompts(self, points, labels, pad=False):
        points = points + 0.5  # Shift to center of pixel
        if pad:
            points = np.pad(points, [[0, 0], [0, 1], [0, 0]])
            labels = np.pad(labels, [[0, 0], [0, 1]], constant_values=-1)
        point_embedding = self.normalize_coords(points)  # [1, 1, 2] -> [1, 1, 256]
        point_embedding[labels == -1] = self.empty_points  # [TODO] labels == -1 is all padded?
        point_embedding[labels == 0] += self.points_height
        point_embedding[labels == 1] += self.points_width
        return point_embedding

    def _embed_boxes_prompts(self, boxes):
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape([-1, 2, 2])
        corner_embedding = self.normalize_coords(coords)
        corner_embedding[:, 0, :] += self.bboxes_height
        corner_embedding[:, 1, :] += self.bboxes_width
        return corner_embedding

    def _embed_masks_prompts(self, masks):
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def preprocess_image(self, image):
        height, width = iamge.shape[:2]
        scale = min(self.input_image_size[0] / height, self.input_image_size[1] / width)
        scale_height, scale_width = int(height * scale + 0.5), int(width * scale + 0.5)




    def __call__(self, image, points=None, labels=None, boxes=None, masks=None):
        image
        batch_size, embed_dim = 1, 320
        points_inputs = self._embed_points_prompts(points, labels, pad=(boxes is None)) if points is not None else np.empty([batch_size, 0, 2])
        boxes_inputs = self._embed_boxes_prompts(boxes) if boxes is not None else np.empty([batch_size, 0, 2, 2])
        masks_inputs = self._embed_masks_prompts(masks) if masks is not None else np.empty([batch_size, 0, 256, 256, 1])
        sparse_embeddings = functional.concat([points_inputs, boxes_inputs])

        src = image_embeddings + masks_inputs

        low_res_masks, iou_predictions = mask_decoder([image_embeddings, sparse_embeddings, self.positional_encoding_gaussian_matrix])
        if not return_logits:
            masks = masks > mask_threshold

        masks_np = masks[0].detach().cpu().numpy()
        iou_predictions_np = iou_predictions[0].detach().cpu().numpy()
        low_res_masks_np = low_res_masks[0].detach().cpu().numpy()
        return masks_np, iou_predictions_np, low_res_masks_np


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
