from enum import Enum


class AminoAcid(Enum):
    '''Amino acid codes, according to
    https://en.wikipedia.org/wiki/FASTA_format'''

    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5
    G = 6
    H = 7
    I = 8  # noqa
    J = 9  # not part of IUB/IUPAC Amino Acid Codes
    K = 10
    L = 11
    M = 12
    N = 13
    P = 14
    Q = 15
    R = 16
    S = 17
    T = 18
    U = 19  # not part of IUB/IUPAC Amino Acid Codes
    V = 20
    W = 21
    Y = 22
    Z = 23

    NUM_AMINO_ACIDS = 24
