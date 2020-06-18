#!/usr/bin/env python
# coding: utf-8

# # Tensorflow Probability & Treeflow Demo
# Author: Christiaan Swanepoel

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import treeflow.beagle
import treeflow.substitution_model
import treeflow.sequences
import treeflow.tree_processing
import treeflow.model
from treeflow.coalescent import ConstantCoalescent
tfd = tfp.distributions

DATA_ROOT = "../../data/"
NEWICK_FILE = DATA_ROOT + "wnv/wnv_seed_6.nwk"
FASTA_FILE = DATA_ROOT + "wnv/wnv.fasta"

tree, taxon_names = treeflow.tree_processing.parse_newick(NEWICK_FILE)
topology = treeflow.tree_processing.update_topology_dict(tree["topology"])
taxon_count = len(taxon_names)
sampling_times = tf.convert_to_tensor(tree["heights"][:taxon_count], dtype=tf.float32)

prior = tfd.JointDistributionNamed(
    dict(
        clock_rate=tfd.LogNormal(loc=0.0, scale=3.0),
        pop_size=tfd.LogNormal(0.0, 3.0),
        tree=lambda pop_size: ConstantCoalescent(taxon_count, pop_size, sampling_times),
    )
)

q_prior = treeflow.model.construct_prior_approximation(prior)
q_tree = treeflow.model.construct_tree_approximation(NEWICK_FILE)
q = tfd.JointDistributionNamed(dict(tree=q_tree, **q_prior))
q.sample()

subst_model = treeflow.substitution_model.HKY()
likelihood, instance = treeflow.beagle.log_prob_conditioned_branch_only(
    FASTA_FILE,
    subst_model,
    # @Christiaan I assume that your convention regarding NT order is the same as that
    # of BEAST.
    frequencies=np.array(
        [
            0.26744051135162256,
            0.22286688874964067,
            0.2787013207712062,
            0.23099127912752943,
        ]
    ),
    kappa=14.52346114599242,
    rescaling=True,
    newick_file=NEWICK_FILE,
)

wrapped_likelihood = lambda z: likelihood(
    treeflow.sequences.get_branch_lengths(z["tree"])
    * tf.expand_dims(z["clock_rate"], -1)
)
wrapped_likelihood(q.sample())


log_posterior = lambda **z: (prior.log_prob(z) + wrapped_likelihood(z))
log_posterior(**q.sample())


# tf.profiler.experimental.start('logdir')
loss = tfp.vi.fit_surrogate_posterior(
    log_posterior, q, tf.optimizers.Adam(learning_rate=0.0001), 50
)
# tf.profiler.experimental.stop()
pd.Series(loss).to_csv("loss.csv", header=False, index=False)
