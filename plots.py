"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from matplotlib.lines import Line2D
from experiments import mnist_lossy, jsonmaps, toy_multisets
import mnist_trial

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from trial import exp
init_trial = exp()
data  = init_trial.values()

#mnist, metrics2 = mnist_trial.run_all_experiments(seed=0)
#print(mnist['compressed_length_multiset'])
#print(mnist['compressed_bits_for_run_length_encoding'])

#exit(0)
#fig, ax = plt.subplots()
#ax.plot(list(mnist.index.values), list(mnist['compressed_length_multiset']), color = "blue")
#ax.set_xlabel("size of multiset")
#ax.set_ylabel("bits")
#ax.set_yscale('log', base=2)
# ax.legend("seq_length vs length of multiset")
# fig.savefig("seq_length_vs_compressed_length_multiset.png")
#ax.plot(list(mnist.index.values), list(mnist['compressed_bits_for_run_length_encoding']), color = "red")
# ax.xlabel("size of multiset")
# ax.ylabel("bits")
# ax.legend("seq_length vs length of multiset")
#fig.savefig("seq_length_vs_compressed_length_multiset.png")
#plt.show()

#exit(0)
# jsons = jsonmaps.run_all_experiments(seed=0)
toyms = toy_multisets.run_all_experiments(seed=0)
# def plot(s, ax, name, **params):
#     ax.plot(s.index, s.avg, label=name, color='k', **params)
#     ax.fill_between(s.index, s.lower, s.upper, alpha=0.25, color='gray')
# fig,ax = plt.subplots()
# print(toyms
#     .loc[:2000]
#     .compressed_length_multiset
#         .unstack('alphabet_size')
#         .swaplevel(0, 1, axis=1)[1024].avg)
# exit(1)

# # Bit savings plot
#  fig, ax = plt.subplots()

# jsons[['saved_bits_limit']]\
# .plot(ax=ax, style=['k--'])

# jsons[['saved_bits']]\
# .loc[2:]\
# .plot(ax=ax, style=['k.-'])

#  mnist[['saved_bits']]\
#  .loc[50:] \
#  .plot(ax=ax, style=['k^-'])

# jsons[['saved_bits_limit_unnested']]\
# .plot(ax=ax, style=['k--'])

# ax.set_xlabel(r'Multiset size $|\mathcal{M}|$')
# ax.set_yscale('log', base=10)
# ax.set_xscale('log', base=10)
# ax.set_yticks([8e1, 8e2, 8e3, 80e3])
# ax.set_ylabel('Savings (bits)')
# ax.set_yticklabels(['10 B', '100 B', '1 kB', '10 kB'])
# ax.autoscale(tight=True)
# ax.set_ylim(5*10)

# ax.legend(['Theoretical limit', 'JSON maps', 'MNIST'], handlelength=1.0)

# fig.savefig('figures/mnist-jsonmaps-savings.pdf')
# fig.savefig('figures/mnist-jsonmaps-savings.jpg', dpi=300)
# plt.show()

# # Bit savings plot percentage
# fig, ax = plt.subplots()

# mnist[['pct_saved_bits_limit']]\
# .plot(ax=ax, style=['k--'])

# jsons[['pct_saved_bits']]\
# .loc[2:]\
# .plot(ax=ax, style=['k.-'])

# mnist[['pct_saved_bits']]\
# .loc[20:]\
# .plot(ax=ax, style=['k^-'])

# jsons[['pct_saved_bits_limit']]\
# .plot(ax=ax, style=['k--'])

# ax.set_xscale('log', base=10)
# ax.set_ylabel('Savings (\%)')
# ax.set_xlabel(r'Multiset size $|\mathcal{M}|$')
# ax.autoscale(tight=True)
# ax.set_ylim(-0.1, 1.5)

# ax.legend(['Theoretical limit', 'JSON maps', 'MNIST'], handlelength=1.0)

# fig.savefig('figures/mnist-jsonmaps-savings-pct.pdf')
# fig.savefig('figures/mnist-jsonmaps-savings-pct.jpg', dpi=300)
# plt.show()

#######################
# Toy multisets plots #
#######################

def plot(index,avg, ax, name,colors, **params):
    ax.plot(index, avg, label=name,color=colors, **params)
    # ax.fill_between(s.index, s.lower, s.upper, alpha=0.25, color='gray')


# # Computational complexity
# fig, ax = plt.subplots()

# for seq_length in [2048, 1024, 512]:
#     times = (toyms
#                 .duration
#                 .unstack('seq_length')
#                 .swaplevel(0, 1, axis=1)
#                 [seq_length])
#     times.pipe(plot, ax=ax, name=seq_length, linestyle='-')
#     ax.text(times.index[1],
#             times['avg'].iloc[-1]+0.05,
#             r'$|\mathcal{M}| = '+ str(seq_length) + r'$',
#             fontsize=10)

# ax.set_xlabel(r'Alphabet size $|\mathcal{A}|$')
# ax.set_ylabel('Time (seconds)')
# ax.set_xscale('log', base=2)
# ax.autoscale(tight=True)
# ax.set_ylim(0, 1.2)
# ax.set_yticks(np.arange(0.2, 1.2, 0.2))

# fig.savefig('figures/toy-multisets-complexity.pdf')
# fig.savefig('figures/toy-multisets-complexity.jpg', dpi=300)
# plt.show()

# Rate plot
fig, ax = plt.subplots()

bits = (toyms
        .loc[:2000]
        .compressed_length_multiset
        .unstack('alphabet_size')
        .swaplevel(0, 1, axis=1))
x = [10, 13, 17]
for i in range(len(x)):
        alph_bits = bits[2**x[i]]
        a = alph_bits.index
        b = alph_bits.avg
        plot(a,b,ax, name='facebook',colors = 'k')
        ax.text(alph_bits.index[-1]-275,
                0.995*alph_bits['avg'].iloc[-1],
                r'$|\mathcal{A}| = 2^{'+ str(x[i]) + r'}$',
                fontsize=10)
x = [10, 13, 17]
for i in range(len(x)):
      alph_bits = bits[2**x[i]]
      a = alph_bits.index
      plot(a,data[i],ax,name='robin',colors='r')
      print(data[i])
    

ax.legend(handles=[
    Line2D([0], [0], label='Compressed size', color='k', linestyle='-'),
    Line2D([0], [0], label=r'Trial', color='r', linestyle='--')],
    loc='upper left',
    handlelength=1,
    handletextpad=0.3)

ax.set_xlabel(r'Multiset size $|\mathcal{M}|$')
ax.set_ylabel('Bits')
ax.set_yscale('log', base=2)
ax.set_yticks(2**np.arange(9, 15))
ax.autoscale(tight=True)
ax.set_ylim(top=2**15)

# fig.savefig('figures/toy-multisets-rate.pdf')
fig.savefig('figures/toy-multisets-rate.jpg', dpi=300)
plt.show()
