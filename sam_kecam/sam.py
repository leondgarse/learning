import math
import numpy as np
from PIL import Image
import keras_cv_attention_models
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format, initializers
from keras_cv_attention_models.attention_layers import BiasLayer, PureWeigths
from keras_cv_attention_models.models import register_model, FakeModelWrapper

def MaskDownscaler(embed_dim=256, mask_in_chans=16):
    return models.Sequential([
        layers.Input([None, None, 1]),
        layers.Conv2D(mask_in_chans // 4, kernel_size=2, strides=2, name="1_conv"),
        layers.LayerNormalization(name="1_ln"),
        layers.Activation(activation="gelu"),
        layers.Conv2D(mask_in_chans, kernel_size=2, strides=2, name="2_conv"),
        layers.LayerNormalization(name="2_ln"),
        layers.Activation(activation="gelu"),
        layers.Conv2D(embed_dim, kernel_size=1, name="output_conv"),
    ], name="mask_downscaler")


def EmptyMask(embed_dim=256):
    return models.Sequential([layers.Input([]), PureWeigths(shape=[1, 256], name="empty_masks")], name="empty_mask")


def BboxesEncoder(embed_dim=256):
    embeddings_initializer = initializers.RandomNormal(mean=0, stddev=1)
    return models.Sequential([layers.Input([2, embed_dim]), BiasLayer(axis=-1, initializer=embeddings_initializer, name="bboxes_bias")], name="bboxes_encoder")


def PointsEncoder(embed_dim=256):
    points = layers.Input([None, embed_dim])
    labels = layers.Input([None], dtype="int64")
    nn = layers.Embedding(3, embed_dim, name="points")(labels + 1)  # labels [-1, 0, 1] -> [0, 1, 2]
    nn = points + nn
    return models.Model([points, labels], nn, name="points_encoder")


def ImageEncoder(input_shape=(1024, 1024, 3), embed_dim=256):
    image_encoder = keras_cv_attention_models.models.TinyViT_5M(input_shape=input_shape, strides=[1, 2, 2, 1], num_classes=0)
    inputs = image_encoder.inputs[0]
    nn = image_encoder.outputs[0]
    nn = layers.Conv2D(embed_dim, kernel_size=1, use_bias=False, name="neck_1_conv")(nn)
    nn = layers.LayerNormalization(name="neck_1_ln")(nn)
    nn = layers.ZeroPadding2D((1, 1))(nn)
    nn = layers.Conv2D(embed_dim, kernel_size=3, use_bias=False, name="neck_2_conv")(nn)
    nn = layers.LayerNormalization(name="neck_2_ln")(nn)
    return models.Model(inputs, nn, name="image_encoder")

class SAM(FakeModelWrapper):  # FakeModelWrapper providing save / load / cuda class methods
    def __init__(self, embed_dim=256, image_input_shape=(1024, 1024), mask_in_chans=16, name="sam"):
        self.image_input_shape = image_input_shape[:2] if isinstance(image_input_shape, (list, tuple)) else image_input_shape
        self.embed_dim = embed_dim

        self.image_encoder = ImageEncoder(input_shape=[*image_input_shape, 3], embed_dim=embed_dim)
        self.points_encoder = PointsEncoder(embed_dim=embed_dim)
        self.bboxes_encoder = BboxesEncoder(embed_dim=embed_dim)
        self.mask_downscaler = MaskDownscaler(embed_dim=embed_dim, mask_in_chans=mask_in_chans)
        self.empty_masks_model = EmptyMask(embed_dim=embed_dim)
        super().__init__([self.image_encoder, self.points_encoder, self.bboxes_encoder, self.mask_downscaler, self.empty_masks_model], name=name)

        scale, num_pos_feats = 1.0, 64
        scale = 1.0 if scale is None or scale <= 0.0 else scale
        self.positional_encoding_gaussian_matrix = scale * np.random.normal(size=[2, num_pos_feats]).astype("float32")

        self.image_embedding_shape = self.image_encoder.output_shape[1:-1] if image_data_format() == "channels_last" else self.image_encoder.output_shape[2:]
        self.empty_points = np.empty([1, 0, self.embed_dim])
        self.empty_bboxes = np.empty([1, 0, self.embed_dim])
        self.empty_masks = self.empty_masks_model.get_layer("empty_masks").get_weights()[0]

        grid = np.ones(self.image_embedding_shape, dtype="float32")
        grid = np.stack([grid.cumsum(axis=0) - 0.5, grid.cumsum(axis=1) - 0.5], axis=-1)
        self.grid_positional_embedding = self.normalize_coords(grid)  # [height, width, 2]

    def normalize_coords(self, coords):
        coords = coords / [self.image_input_shape[1], self.image_input_shape[0]] # [TODO] 0, 1 ???
        coords = (2 * coords - 1) * (2 * np.pi)
        coords = coords @ self.positional_encoding_gaussian_matrix  # [1, 1, 2] @ [2, 128] -> [1, 1, 128]
        return np.concatenate([np.sin(coords), np.cos(coords)], axis=-1)  # [1, 1, 256]

    def preprocess_image(self, image, scaled_height, scaled_width):
        """ Aspect awared image resize -> Normalize to ~[-2, 2] -> pad to self.image_input_shape """
        image = np.array(Image.fromarray(image).resize([scaled_height, scaled_width]))[None]
        mean, std = np.array([123.675, 116.28, 103.53]).astype("float32"), np.array([58.395, 57.12, 57.375]).astype("float32")
        normed_image = (image - mean) / std
        pad_height, pad_width = self.image_input_shape[0] - normed_image.shape[0], self.image_input_shape[1] - normed_image.shape[1]
        padded_image = np.pad(normed_image, [[0, pad_height], [0, pad_width], [0, 0]])[None]
        return padded_image

    def preprocess_points(self, points, height_scale, width_scale, pad=False):
        points = (points.reshape([1, -1, 2]) * [height_scale, width_scale])
        points = points + 0.5  # Shift to center of pixel
        points = np.pad(points, [[0, 0], [0, 1], [0, 0]]) if pad else points
        return self.normalize_coords(points)

    def preprocess_boxes(self, boxes, height_scale, width_scale):
        boxes = (boxes.reshape([1, -1, 2, 2]) * [height_scale, width_scale])
        boxes = boxes + 0.5  # Shift to center of pixel
        return self.normalize_coords(boxes)

    def __call__(self, image, points=None, labels=None, boxes=None, masks=None):
        points, labels, boxes, mask_inputs, return_logits, mask_threshold = np.array([[400, 400]]), np.array([1]), None, None, False, 0.0

        orign_height, orign_width = image.shape[:2]
        scale = min(self.image_input_shape[0] / orign_height, self.image_input_shape[1] / orign_width)
        scaled_height, scaled_width = int(orign_height * scale + 0.5), int(orign_width * scale + 0.5)
        height_scale, width_scale = scaled_height / image_height, scaled_width / image_width

        image_embeddings = self.image_encoder(self.preprocess_image(image, scaled_height, scaled_width))
        boxes_inputs = self.bboxes_encoder(self.preprocess_boxes(boxes, height_scale, width_scale)) if boxes is not None else self.empty_bboxes
        if points is not None and labels is not None:
            pad = boxes is None
            points = self.preprocess_points(points, height_scale, width_scale, pad=pad)
            labels = labels.reshape([1, -1])
            labels = np.pad(labels, [[0, 0], [0, 1]], constant_values=-1) if pad else labels
            points_inputs = self.points_encoder([points, labels])
        else:
            points_inputs = self.empty_points
        sparse_embeddings = functional.concat([points_inputs, boxes_inputs])

        masks_inputs = self.mask_downscaler(masks) if masks is not None else self.empty_masks
        image_with_masks_inputs = image_embeddings + masks_inputs
        low_res_masks, iou_predictions = mask_decoder([image_with_masks_inputs, sparse_embeddings, self.grid_positional_embedding])
        low_res_masks = low_res_masks.cpu().numpy() if hasattr(low_res_masks, "cpu") else low_res_masks.numpy()
        iou_predictions = iou_predictions.cpu().numpy() if hasattr(iou_predictions, "cpu") else iou_predictions.numpy()

        """ Remove padding and upscale masks to the original image size. """
        masks = backend.numpy_image_resize(low_res_masks, self.image_input_shape, method="bilinear")
        masks = masks[:, :scaled_height, :scaled_width] if image_data_format() == "channels_last" else masks[:, :, :scaled_height, :scaled_width]
        masks = backend.numpy_image_resize(masks, [orign_height, orign_width], method="bilinear")
        masks = masks if return_logits else (masks > mask_threshold)
        return masks, iou_predictions, low_res_masks
