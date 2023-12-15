
from multiset_codec import codecs
from multiset_codec.msbst import build_multiset
from scipy.stats import dirichlet
from time import time

import numpy as np
import math


class exp():
    def __int__(self):
        return None
    def values(self):
        seq_lens = [512,645,812,1024,1290,1625]
        alphabet_sizes=[2**10,2**13,2**17]
        alphabet_op = []
        for k in range(len(alphabet_sizes)):
            temp_list = []
            for j in range(len(seq_lens)):
                seed=1337
                np.random.seed(seed)
                alphabet_size = alphabet_sizes[k]
                seq_length = seq_lens[j]
                # Sample skewed source from a Dirichlet distribution.
                alphabet = np.arange(alphabet_size)
                source_probs = dirichlet.rvs(alphabet+1).flatten()


                # Create a subset of the alphabet, called alphabet_seen, that contains
                # 512 unique symbols. Only symbols from alphabet_seen will appear in the
                # multiset. This is required to show that the complexity will not scale
                # with alphabet size.
                alphabet_seen = np.random.choice(
                        alphabet, size=10, p=source_probs, replace=False)

                source_probs_seen = source_probs[alphabet_seen]
                source_probs_seen /= source_probs_seen.sum()

                # Sample a random sequence, but guarantee exactly 512 unique symbols.
                # To do this, we start with alphabet_seen, and append seq_length-512
                # symbols sampled from alphabet_seen.
                sequence = np.r_[alphabet_seen, np.random.choice(
                        alphabet_seen, size=seq_length-512, p=source_probs_seen)]

                # The symbols will be encoded to the ANS state with a codec
                # that has the exact source probabilities. Note that source_probs
                # has size len(alphabet) > len(alphabet_seen) = 512, hence it scales
                # with alphabet size. However, the codec implementation is efficient,
                # and the dependency does not manifest in these experiments.


                # Start timing, to estimate compute time
                time_start = time()

                # Populate multiset via successive applications of the insert operation.
                multiset = build_multiset(sequence)
                x = sequence
                # print(sequence.shape)
                def forward_lookup(multiset, x):
                    '''
                    Looks up the cumulative (start) and frequency (freq) counts of symbol x.
                    '''
                    if not multiset:
                        raise ValueError("The symbol {} could not be found.".format(x))
                    size, y, left, right = multiset
                    if x > y:
                        start_right, freq = forward_lookup(right, x)
                        start = size - right[0] + start_right
                    elif x < y:
                        start, freq = forward_lookup(left, x)
                    else:
                        start = left[0] if left else 0
                        freq = size - start - (right[0] if right else 0)
                    return start, freq

                dict = {}
                for i in range(len(x)):
                    _ ,f = forward_lookup(multiset, x[i])
                    dict[x[i]] = f
                sum = 0
                val = list(dict.values())
                log2comp = []
                for i in range(len(val)):
                    temp = math.ceil(math.log2(val[i]))
                    log2comp.append(temp)
                log2comp = np.array(log2comp)
                sumlogs = np.sum(log2comp)
                bits_symbol=math.ceil(math.log2(alphabet_sizes[k]))
                temp_list.append(sumlogs*bits_symbol)
            alphabet_op.append(temp_list)
        return alphabet_op

