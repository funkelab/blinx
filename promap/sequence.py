from .amino_acids import AminoAcid
import numpy as np


def count_amino_acids(sequence):
    '''Get a count of amino acids for the given sequence. Ignores amino
    acid code ``X`` ("any amino acid").

    Args:

        sequence (string):
            The sequence of the protein as a string, e.g.,
            ``MIESENLNQE...``.

    Returns:

        An array mapping :class:`AminoAcid` (or more specifically its value) to
        ``int`` (the number of occurrences).
    '''

    sequence = sequence.upper()
    sequence = sequence.replace('X', '')
    sequence = np.frombuffer(bytes(sequence, 'ascii'), dtype=np.uint8)

    amino_acids, counts = np.unique(sequence, return_counts=True)
    amino_acid_indices = [
        AminoAcid[a].value
        for a in amino_acids.tobytes().decode('ascii')
    ]

    total_counts = np.zeros((AminoAcid.NUM_AMINO_ACIDS.value,), dtype=np.int32)
    total_counts[amino_acid_indices] = counts

    return total_counts
