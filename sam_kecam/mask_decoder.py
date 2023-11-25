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


def attention(query, key=None, value=None, num_heads=4, head_dim=0, name=""):
    _, bb, cc = query.shape
    head_dim = head_dim if head_dim > 0 else cc // num_heads
    emded_dim = int(num_heads * head_dim)

    query = layers.Dense(emded_dim, use_bias=False, name=name and name + "query")(query)
    key = layers.Dense(emded_dim, use_bias=False, name=name and name + "key")(query if key is None else key)
    value = layers.Dense(emded_dim, use_bias=False, name=name and name + "value")(query if value is None else value)

    query, key, value = qkv_to_multi_head_channels_last_format(query, key, value, num_heads, data_format="channels_last")
    output_shape = [-1, -1 if bb is None else bb, cc]
    return scaled_dot_product_attention(query, key, value, output_shape, out_weight=True, out_bias=True, name=name)


def two_way_attention_block(
    query, key, query_position, key_position, num_heads=4, downsample_rate=2, skip_first_layer_pe=False, mlp_ratio=8, activation="relu", name=""
):
    if skip_first_layer_pe:
        query = attention(query, downsample_rate=downsample_rate, name=name + "query_")
    else:
        attn_out = attention(query + query_position, value=query, name=name + "query_")
        query = query + attn_out
    query = layer_norm(query, name=name + "query_")

    # Cross attention block, tokens attending to image embedding
    query_with_pe = query + query_position
    key_with_pe = key + key_position
    head_dim = query.shape[-1] // num_heads // downsample_rate
    attn_out = attention(query_with_pe, key=key_with_pe, value=key, num_heads=num_heads, head_dim=head_dim, name=name + "cross_embedding_")
    query = query + attn_out
    query = layer_norm(query, name=name + "cross_embedding_")

    # MLP block
    mlp_out = mlp_block(query, hidden_dim=int(query.shape[-1] * mlp_ratio), activation=activation, name=name + "mlp_")
    query = query + mlp_out
    query = layer_norm(query, name=name + "mlp_")

    # Cross attention block, image embedding attending to tokens
    query_with_pe = query + query_position
    attn_out = attention(key_with_pe, key=query_with_pe, value=query, head_dim=head_dim, name=name + "cross_tokens_")
    key = key + attn_out
    key = layer_norm(key, name=name + "cross_tokens_")
    return query, key


def two_way_transformer(
    image_embedding, image_position, point_embedding, depth=2, num_heads=8, mlp_dim=2048, downsample_rate=2, mlp_ratio=8, activation="relu", name=""
):
    query, key, query_position, key_position = point_embedding, image_embedding, point_embedding, image_position
    for ii in range(depth):
        skip_first_layer_pe = True if ii == 0 else False
        query, key = two_way_attention_block(
            query, key, query_position, key_position, num_heads, downsample_rate, skip_first_layer_pe, mlp_ratio, activation, name=name + "{}_".format(ii)
        )

    query_with_pe = query + query_position
    key_with_pe = key + key_position
    head_dim = query.shape[-1] // num_heads // downsample_rate
    attn_out = attention(query_with_pe, key=key_with_pe, value=key, num_heads=num_heads, head_dim=head_dim, name=name + "token_to_image_")
    query = query + attn_out
    query = layer_norm(query, name=name + "token_to_image_")
    return query, key


def MaskDecoder(embed_dims=256, num_mask_tokens=4, name="mask_decoder"):
    image_embedding = layers.Input([None, None, embed_dims], batch_size=1)
    point_embedding = layers.Input([None, embed_dims], batch_size=1)
    image_position = layers.Input([None, None, embed_dims], batch_size=1)

    point_embedding_with_tokens = ClassToken(num_tokens=5, name="cls_token")(point_embedding)
    iou_masks, encoded_image_embedding = two_way_transformer(image_embedding, image_position, point_embedding_with_tokens, name="attn_")

    # output_upscaling
    encoded_image_embedding = layers.ConvTranspose2d(embed_dims // 4, kernel_size=2, strides=2, name="up_1_conv_transpose")(encoded_image_embedding)
    encoded_image_embedding = layer_norm(encoded_image_embedding)
    encoded_image_embedding = activation_by_name(encoded_image_embedding, activation=activation)
    encoded_image_embedding = layers.ConvTranspose2d(embed_dims // 8, kernel_size=2, strides=2, name="up_2_conv_transpose")(encoded_image_embedding)
    upscaled_image_embedding = activation_by_name(encoded_image_embedding, activation=activation)

    iou_token_out, masks_top, masks_left, masks_bottom, masks_right = functional.unstack(iou_masks, axis=1)
    hyper_in = []
    for id, (ii, name) in enumerate(zip([masks_top, masks_left, masks_bottom, masks_right], ["top", "left", "bottom", "right"])):
        hyper_in.append(mlp_block(embed_dims, hidden_dim=embed_dims, output_channel=embed_dims, activation=activation, name="masks_" + name))
    iou_pred = mlp_block(iou_token_out)
    hyper_in = [ for ]
    hyper_in = functional.stack(hyper_in, axis=1)
    masks = hyper_in @ upscaled_image_embedding

    return models.Model([image_with_masks_inputs, sparse_prompt_embedding, image_position_embedding], [masks, iou_pred], name=name)
