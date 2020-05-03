import sys
import matplotlib
import matplotlib.pyplot as plt

import jax
import jax.numpy as np
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

import pandas as pd

from covid.models.SEIR_hierarchical import SEIR_hierarchical
import covid.util as util

if __name__ == "__main__":
     states = ['MA', 'NY','WA','TX',"ID","AL","SD"]
     data, place_data = util.load_state_Xy()
     data = data.join(place_data.drop(columns='state'), how='inner')

     args = {
         'data': data,
        'place_data' : place_data,
          'use_rw': False,
         'rw_scale': 1e-2,
          'det_noise_scale' : 0.2
     }

     prob_model = SEIR_hierarchical

     kernel = NUTS(prob_model,
              init_strategy = numpyro.infer.util.init_to_median())

     mcmc = MCMC(kernel, 
            num_warmup=1000,
            num_samples=1000,
            num_chains=1)

     mcmc.run(jax.random.PRNGKey(1), use_obs=True, **args)

     mcmc.print_summary()
     mcmc_samples = mcmc.get_samples() 

     prior = Predictive(prob_model, posterior_samples = {}, num_samples = 100)
     prior_samples = prior(PRNGKey(2), **args)
     args['rw_scale'] = 0 # set drift to zero for forecasting
     post_pred = Predictive(prob_model, posterior_samples = mcmc_samples)
     post_pred_samples = post_pred(PRNGKey(2), T_future=100, **args)


     util.write_summary('US_covariates_large_var_bigger_det_rate', mcmc)
     util.save_samples('US_covariates_large_var_bigger_det_rate', prior_samples, mcmc_samples, post_pred_samples)
