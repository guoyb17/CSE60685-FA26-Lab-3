"""Complete channel selection and weight transfer, supplied for Lab 3."""
import torch


def select_conv2_channels(weight, keep_count):
    scores = weight.abs().sum(dim=(1, 2, 3))
    ranked = torch.argsort(scores, descending=True, stable=True)
    return ranked[:keep_count].sort().values


def copy_compact_weights(source, compact, keep):
    src_f, dst_f = source.features, compact.features
    src_c, dst_c = source.classifier, compact.classifier
    with torch.no_grad():
        dst_f[0].load_state_dict(src_f[0].state_dict())
        dst_f[3].weight.copy_(src_f[3].weight[keep])
        dst_f[3].bias.copy_(src_f[3].bias[keep])
        old_fc = src_c[0].weight.reshape(120, 16, 4, 4)
        kept_fc = old_fc[:, keep, :, :].reshape(120, -1)
        dst_c[0].weight.copy_(kept_fc)
        dst_c[0].bias.copy_(src_c[0].bias)
        dst_c[2].load_state_dict(src_c[2].state_dict())
        dst_c[4].load_state_dict(src_c[4].state_dict())
