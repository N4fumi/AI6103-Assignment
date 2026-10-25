def make_transform(cfg, stats, training):
    operations = [T.Resize((cfg['image_size'], cfg['image_size']), interpolation=T.InterpolationMode.BILINEAR, antialias=True)]
    if training:
        operations += [T.Pad(cfg['padding'], fill=0, padding_mode='constant'),
                       T.RandomCrop(cfg['image_size']), T.RandomHorizontalFlip(cfg['flip_probability'])]
    return T.Compose(operations + [T.ToTensor(), T.Normalize(stats['mean'], stats['std'])])
