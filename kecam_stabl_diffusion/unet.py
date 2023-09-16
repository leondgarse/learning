import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, initializers, image_data_format
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    group_norm,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {"": {"": ""}}

class SinusoidalTimeStepEmbedding(layers.Layer):
    def __init__(self, hidden_channels=320, max_period=10000, **kwargs):
        super().__init__(**kwargs)
        self.hidden_channels, self.max_period = hidden_channels, max_period

    def build(self, input_shape):
        # input_shape: [batch]
        half = self.hidden_channels // 2  # half the channels are sin and the other half is cos
        frequencies = np.exp(-np.log(self.max_period) * np.arange(0, half, dtype="float32") / half)[None]
        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("frequencies", functional.convert_to_tensor(frequencies, dtype=self.compute_dtype), persistent=False)
        else:
            self.frequencies = functional.convert_to_tensor(frequencies, dtype=self.compute_dtype)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        cur = inputs[:, None] * self.frequencies
        return functional.concat([functional.cos(cur), functional.sin(cur)], axis=-1)

    def compute_output_shape(self, input_shape):
        return [None, self.hidden_channels]

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"hidden_channels": self.hidden_channels, "max_period": self.max_period})
        return base_config

def down_sample(inputs, name=""):
    return inputs

def up_sample(inputs, name=""):
    return inputs

def res_block(inputs, time_embedding, name=""):
    return inputs

def attention_block(inputs, condition, name=""):
    return inputs

def UNet(
    input_shape=(640, 640, 4),
    num_blocks=[2, 2, 2, 2],
    hidden_channels=320,
    hidden_expands=[1, 2, 4, 4],
    num_attention_blocks=[1, 1, 1, 0],  # attention_blocks after each res_block in each stack
    num_heads=8,
    conditional_embedding=768,
    pretrained=None,
    model_name="unet",
    kwargs=None,  # Not using, recieving parameter
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    inputs = layers.Input(backend.align_input_shape_by_image_data_format(input_shape))
    channel_axis = -1 if backend.image_data_format() == "channels_last" else 1
    time_steps = layers.Input([])
    condition = layers.Input([None, conditional_embedding])

    time_embedding = SinusoidalTimeStepEmbedding()(time_steps)
    time_embedding = layers.Dense(hidden_channels * 4, name="time_embed_1_")(time_embedding)
    time_embedding = activation_by_name(time_embedding, activation="swish", name="time_embed_")
    time_embedding = layers.Dense(hidden_channels * 4, name="time_embed_2_")(time_embedding)

    nn = conv2d_no_bias(inputs, hidden_channels, kernel_size=3, padding="SAME", name="latents_")

    """ Down blocks """
    skip_connections = [nn]
    for stack_id, (num_block, hidden_expand, num_attention_block) in enumerate(zip(num_blocks, hidden_expands, num_attention_blocks)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            nn = down_sample(nn)
            skip_connections.append(nn)
        for block_id in range(num_block):
            block_name = stack_name + "down_block{}_".format(block_id + 1)
            nn = res_block(nn, time_embedding, name=block_name)
            for attention_block_id in range(num_attention_block):
                nn = attention_block(nn, condition, name=block_name + "attn_{}_".format(attention_block_id))
            skip_connections.append(nn)
    print(f">>>> {[ii.shape for ii in skip_connections] = }")

    """ Middle blocks """
    nn = res_block(nn, time_embedding, name="middle_block_1_")
    nn = attention_block(nn, condition, name=block_name + "middle_block_attn_")
    nn = res_block(nn, time_embedding, name="middle_block_2_")

    """ Up blocks """
    for stack_id, (num_block, hidden_expand, num_attention_block) in enumerate(zip(num_blocks, hidden_expands, num_attention_blocks)):
        stack_name = "stack{}_".format(len(num_blocks) + stack_id + 1)
        if stack_id > 0:
            nn = up_sample(nn)
        for block_id in range(num_block + 1):
            block_name = stack_name + "up_block{}_".format(block_id + 1)
            skip_connection = skip_connections.pop(-1)
            nn = functional.concat([nn, skip_connection], axis=channel_axis)

            nn = res_block(nn, time_embedding, name=block_name)
            for attention_block_id in range(num_attention_block):
                nn = attention_block(nn, condition, name=block_name + "attn_{}_".format(attention_block_id))

    """ Output blocks """
    output_channels = inputs.shape[channel_axis]
    nn = group_norm(nn, name="output_")
    nn = activation_by_name(nn, activation="swish", name="output_")
    outputs = conv2d_no_bias(nn, output_channels, kernel_size=3, padding="SAME", name="output_")

    model = models.Model(inputs, outputs, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "stable_diffusion", pretrained)
    # add_pre_post_process(model, rescale_mode=rescale_mode, post_process=post_process)
    return model
