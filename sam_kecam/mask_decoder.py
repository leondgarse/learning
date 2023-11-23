import math
import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format, initializers
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    add_with_layer_scale_and_drop_block,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    inverted_residual_block,
    layer_norm,
    mlp_block,
    multi_head_self_attention,
    mhsa_with_multi_head_position,
    window_attention,
    ClassToken,
    MultiHeadPositionalEmbedding,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights


class RandomFrequencyPositionalEmbeddings(keras.layers.Layer):
    """Positional encoding using random spatial frequencies, maps coordinates/points in 2D space to positional encodings using random spatial frequencies."""
    def __init__(self, num_positional_features, scale, **kwargs):
        super().__init__(**kwargs)
        self.num_positional_features, self.scale = num_positional_features, scale

    def build(self, input_shape=None):
        self.height, self.width = input_shape[1:-1] if backend.image_data_format() == "channels_last" else input_shape[2:]
        positional_encoding = np.random.normal(size=(2, self.num_positional_features)).astype("float32")
        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("positional_encoding", functional.convert_to_tensor(positional_encoding, dtype=self.compute_dtype), persistent=False)
        else:
            self.positional_encoding = functional.convert_to_tensor(positional_encoding, dtype=self.compute_dtype)
        super().build(input_shape)

    def call(self, inputs):
        return self.encode_image(inputs)

    def __positional_encodings(self, coords):
        coords = coords * 2 - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = coords * (2 * math.pi)
        return ops.concatenate([ops.sin(coords), ops.cos(coords)], axis=-1)

    def encode_image(self, size):
        """Generate a positional encoding for an image of any given size."""
        H, W = size
        grid = ops.ones(shape=(H, W), dtype=self.dtype)
        y_embed = ops.cumsum(grid, axis=0) - 0.5
        x_embed = ops.cumsum(grid, axis=1) - 0.5
        y_embed = y_embed / ops.cast(H, self.dtype)
        x_embed = x_embed / ops.cast(W, self.dtype)
        return self.__positional_encodings(ops.stack([x_embed, y_embed], axis=-1))

    def encode_coordinates(self, coords_input, image_size):
        """Positionally encode points that are not normalized to `[0, 1]`."""
        coords_normalized = ops.stack([coords_input[..., 0] / image_size[1], coords_input[..., 1] / image_size[0]], axis=-1)
        return self.__positional_encodings(coords_normalized)

    def get_config(self):
        config = super().get_config()
        config.update({"num_positional_features": self.num_positional_features, "scale": self.scale})
        return config


def attention(inputs, condition=None, num_heads=4, head_dim=0, name=""):
    _, bb, cc = inputs.shape
    head_dim = head_dim if head_dim > 0 else cc // num_heads
    emded_dim = int(num_heads * head_dim)
    condition = inputs if condition is None else condition

    query = layers.Dense(emded_dim, use_bias=False, name=name and name + "query")(inputs)
    key = layers.Dense(emded_dim, use_bias=False, name=name and name + "key")(condition)
    value = layers.Dense(emded_dim, use_bias=False, name=name and name + "value")(condition)

    query, key, value = qkv_to_multi_head_channels_last_format(query, key, value, num_heads, data_format="channels_last")
    output_shape = [-1, -1 if bb is None else bb, cc]
    return scaled_dot_product_attention(query, key, value, output_shape, out_weight=True, out_bias=True, name=name)


def two_way_attention_block(query, key, query_position, key_position, downsample_rate=2, skip_first_layer_pe=False, name=""):
    if skip_first_layer_pe:
        query = attention(query, downsample_rate=downsample_rate, name=name)
    else:
        attn_out = attention(query + query_position, value=query, name=name)
        query = query + attn_out
    query = layer_norm(query)

    # Cross attention block, tokens attending to image embedding
    query_with_pe = query + query_position
    key_with_pe = key + key_position
    attn_out = attention(query_with_pe, key=key_with_pe, value=key, downsample_rate=downsample_rate, name=name)
    query = query + attn_out
    query = layer_norm(query)

    # MLP block
    mlp_out = mlp_block(query)
    query = query + mlp_out
    query = layer_norm(query)

    # Cross attention block, image embedding attending to tokens
    query_with_pe = query + query_position
    attn_out = attention(key_with_pe, key=query_with_pe, value=query, downsample_rate=downsample_rate, name=name)
    key = key + attn_out
    key = layer_norm(key)
    return query, key


def two_way_transformer(
    image_embedding, image_position, query_position, depth=2, num_heads=8, embedding_dim=256, mlp_dim=2048, downsample_rate=2, activation="relu", name=""
):
    query, key = query_position, image_embedding
    for ii in range(depth):
        query, key = two_way_attention_block(query=query, key=key, query_position=query_position, key_position=image_position, name=name)

    query_with_pe = query + query_positional
    key_with_pe = key + key_positional
    attn_out = attention(query_with_pe, key=key_with_pe, value=key, downsample_rate=downsample_rate, name=name)
    query = query + attn_out
    query = layer_norm(query)
    return query, key


def MaskDecoder(embed_dims=256, num_mask_tokens=4):
    image_embedding = layers.Input([None, None, embed_dims], batch_size=1)
    image_position_embedding = layers.Input([embed_dims], batch_size=1)
    sparse_prompt_embedding = layers.Input([1, embed_dims], batch_size=1)
    dense_prompt_embedding = layers.Input([1, 1, embed_dims], batch_size=1)

    tokens = ClassToken(num_tokens=5)(sparse_prompt_embedding)
    src = image_embedding + dense_prompt_embedding
    iou_masks, src = two_way_transformer(src, image_position_embedding, tokens)

    # output_upscaling
    src = layers.ConvTranspose2d()(src)
    src = layers.LayerNormalization()(src)
    src = activation_by_name(src, activation=activation)
    src = layers.ConvTranspose2d()(src)
    upscaled_embedding = activation_by_name(src, activation=activation)

    iou_token_out, masks_top, masks_left, masks_bottom, masks_right = functional.unstack(iou_masks, axis=1)
    iou_pred = mlp_block(iou_token_out)
    hyper_in = [mlp_block(ii) for id, ii in enumerate([masks_top, masks_left, masks_bottom, masks_right])]
    hyper_in = functional.stack(hyper_in, axis=1)
    masks = hyper_in @ upscaled_embedding

    return masks, iou_pred
