CLASS_NUMBER = {
    'CIFAR10' : 10,
    'CIFAR10-C' : 10,
    'CIFAR100' : 100,
    'Melanoma' : 7,
    'SVHN' : 10,
    'GTSRB' : 43,
    'Flowers102' : 102,
    'DTD' : 47,
    'Food101' : 101,
    'EuroSAT' : 10,
    'OxfordIIITPet' : 37,
    'UCF101' : 101,
    'FMoW' : 62,
    'ImageNet1k' : 1000,
}

MAP_NUMBER = {
    'CIFAR10' : [1, 5, 10],
    'CIFAR10-C' : [1, 5, 10],
    'CIFAR100' : [1, 5, 10],
    'Melanoma' : [1, 5, 10],
    'SVHN' : [1, 5, 10],
    'GTSRB' : [1, 5, 10],
    'Flowers102' : [1, 5, 9],
    'DTD' : [1, 5, 10],
    'Food101' : [1, 5, 9],
    'EuroSAT' : [1, 5, 10],
    'OxfordIIITPet' : [1, 5, 10],
    'UCF101' : [1, 5, 9],
    'FMoW' : [1, 5, 10],
    'ImageNet1k' : [1],
}

IMG_SIZE = {
    'CIFAR10' : 128,
    'CIFAR10-C' : 128,
    'CIFAR100' : 128, 
    'Melanoma' : 128,
    'SVHN' : 128,
    'GTSRB' : 128,
    'Flowers102' : 128,
    'DTD' : 128,
    'Food101' : 128,
    'EuroSAT' : 128,
    'OxfordIIITPet' : 128,
    'UCF101' : 128,
    'FMoW' : 128,
    'ImageNet1k' : 128,
}

SOURCE_CLASS_NUM = {
    'resnet18' : 1000,
    'vit_b_16' : 1000,
    'clip' : 81, 
    'swin_t' : 1000,
    'ig_resnext101_32x8d' : 1000,
}

BATCH_SIZE = {
    'CIFAR10' : 128,
    'CIFAR10-C' : 128,
    'CIFAR100' : 32,
    'Melanoma' : 128,
    'SVHN' : 64,
    'GTSRB' : 128,
    'Flowers102' : 64,
    'DTD' : 32,
    'Food101' : 64,
    'EuroSAT' : 128,
    'OxfordIIITPet' : 128,
    'UCF101' : 128,
    'FMoW' : 128, # 256
    'ImageNet1k' : 128,
}


NETMEAN = {
    'resnet18' : [0.485, 0.456, 0.406],
    'vit_b_16' : [0.485, 0.456, 0.406],
    'clip' : [0.485, 0.456, 0.406], 
    'swin_t' : [0.485, 0.456, 0.406],
    'ig_resnext101_32x8d' : [0.485, 0.456, 0.406],
}

NETSTD = {
    'resnet18' : [0.229, 0.224, 0.225],
    'vit_b_16' : [0.229, 0.224, 0.225],
    'clip' : [0.229, 0.224, 0.225], 
    'swin_t' : [0.229, 0.224, 0.225],
    'ig_resnext101_32x8d' : [0.229, 0.224, 0.225],
}

DEFAULT_TEMPLATE = "This is a photo of a {}."

ENSEMBLE_TEMPLATES = [
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

GTSRB_LABEL_MAP = {
    '0': '20_speed',
    '1': '30_speed',
    '2': '50_speed',
    '3': '60_speed',
    '4': '70_speed',
    '5': '80_speed',
    '6': '80_lifted',
    '7': '100_speed',
    '8': '120_speed',
    '9': 'no_overtaking_general',
    '10': 'no_overtaking_trucks',
    '11': 'right_of_way_crossing',
    '12': 'right_of_way_general',
    '13': 'give_way',
    '14': 'stop',
    '15': 'no_way_general',
    '16': 'no_way_trucks',
    '17': 'no_way_one_way',
    '18': 'attention_general',
    '19': 'attention_left_turn',
    '20': 'attention_right_turn',
    '21': 'attention_curvy',
    '22': 'attention_bumpers',
    '23': 'attention_slippery',
    '24': 'attention_bottleneck',
    '25': 'attention_construction',
    '26': 'attention_traffic_light',
    '27': 'attention_pedestrian',
    '28': 'attention_children',
    '29': 'attention_bikes',
    '30': 'attention_snowflake',
    '31': 'attention_deer',
    '32': 'lifted_general',
    '33': 'turn_right',
    '34': 'turn_left',
    '35': 'turn_straight',
    '36': 'turn_straight_right',
    '37': 'turn_straight_left',
    '38': 'turn_right_down',
    '39': 'turn_left_down',
    '40': 'turn_circle',
    '41': 'lifted_no_overtaking_general',
    '42': 'lifted_no_overtaking_trucks'
}