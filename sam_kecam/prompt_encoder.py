def PointsEncoder(embed_dim=256):
    return lambda xx: initializers.zeros()([xx.shape[0], embed_dim])

def BoxesEncoder(embed_dim=256):
    return lambda xx: initializers.zeros()([xx.shape[0], embed_dim])

def MasksEncoder(embed_dim=256):
    return lambda xx: initializers.zeros()([xx.shape[0], embed_dim])
