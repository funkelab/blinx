from .amino_acids import AminoAcid
from .sequence import count_amino_acids
import json
import logging
import numpy as np
import time

logger = logging.getLogger(__name__)


class ProteinDb:

    def __init__(self, filename):

        logger.info("Reading protein sequence JSON...")
        start = time.time()

        with open(filename, 'r') as f:
            protein_db = json.load(f)

        self.proteins = protein_db['proteins']
        self.num_proteins = len(self.proteins)

        logger.info(
            "Read %d proteins in %.3fs",
            self.num_proteins,
            time.time() - start)

        if self.num_proteins == 0:
            return

        if 'amino_acid_counts' in self.proteins[0]:

            logger.info("Extracting amino-acid counts...")
            start = time.time()

            self.amino_acid_counts = np.array(
                [
                    protein['amino_acid_counts']
                    for protein in self.proteins
                ],
                dtype=np.int32)

            logger.info(
                "Extracted amino acid counts in %.3fs",
                time.time() - start)

        else:

            logger.info("Counting amino-acids...")
            start = time.time()

            self.amino_acid_counts = np.zeros(
                (self.num_proteins, AminoAcid.NUM_AMINO_ACIDS.value),
                dtype=np.int32)
            for i, protein in enumerate(self.proteins):
                self.amino_acid_counts[i] = count_amino_acids(
                    protein['sequence'])

            logger.info(
                "Counted amino acids in %.3fs",
                time.time() - start)

    def get_amino_acid_counts(
            self,
            amino_acids=None,
            num_samples=None):
        '''Get a numpy array of counts for the given amino acids. By default,
        returns these counts for all proteins in the DB. If ``random_sample``
        is ``True``, only ``num_samples`` randomly selected proteins will be
        used.

        Args:

            amino_acids (list of :class:`AminoAcid`, optional):
                The amino acids to get the counts for. If not given, get the
                counts for all amino acids.

            num_samples (int, optional):
                Size of the random sample. If not given, amino acid counts for
                all proteins in the DB are returned.

        Returns:

            A tuple ``(identifiers, counts)``.
        '''

        if amino_acids is not None:
            aa_indices = [aa.value for aa in amino_acids]
        else:
            aa_indices = slice(None)

        if num_samples is not None:
            sample_indices = np.random.choice(
                len(self.proteins),
                num_samples,
                replace=False)
            proteins = [self.proteins[i] for i in sample_indices]
            amino_acid_counts = self.amino_acid_counts[sample_indices]
        else:
            proteins = self.proteins
            amino_acid_counts = self.amino_acid_counts

        identifiers = [protein['identifier'] for protein in proteins]
        counts = amino_acid_counts[:, aa_indices]

        return identifiers, counts
