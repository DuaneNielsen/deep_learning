
from layerbuilder import scan_token

def equal(a, b):
    assert len(a) == len(b)
    for (s0, c0, o0), (s1, c1, o1) in zip(a, b):
        assert s0 == s1
        assert c0 == c1
        assert o0 == o1


def parse_spec(cfg):

    layers = []

    for token in cfg:
        tipe, in_channels, out_channels = scan_token(token)
        layers.append((tipe, in_channels, out_channels))

    return layers


def test_parse():

    cfg = ['C:3:64', 'R:64:64', 'M', 'R:64:128', 'R:128:256', 'M']

    expected = [('C', 3, 64), ('R', 64, 64), ('M', None, None), ('R', 64, 128), ('R', 128, 256), ('M', None, None)]

    equal(expected, expected)

    layers = parse_spec(cfg)

    print('')
    print('')
    print(layers)

    equal(layers, expected)
