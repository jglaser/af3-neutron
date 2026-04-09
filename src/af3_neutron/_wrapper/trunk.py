from typing import Dict, Tuple

import haiku as hk
import jax
import jax.numpy as jnp

from alphafold3.model import model, feat_batch
from alphafold3.model.network import evoformer_network

# -------------------------------------------------------------------------
# HAIKU NAMESPACE WRAPPERS
# These force our detached components to inherit the 'diffuser/' prefix
# so Haiku can find the DeepMind pre-trained weights.
# -------------------------------------------------------------------------
class TrunkWrapper(hk.Module):
    """
    Haiku module that executes the AlphaFold 3 Evoformer trunk.

    This wrapper isolates the Pairformer/Evoformer blocks, managing the
    recycling loop to generate the structural conditionings (single and pair
    embeddings) required by the diffusion head.

    Parameters
    ----------
    config : alphafold3.model.model.Model.Config
        The AlphaFold 3 configuration object.
    name : str, optional
        The Haiku variable scope name, by default 'diffuser'.
    """

    def __init__(self, config: model.Model.Config, name: str = "diffuser"):
        super().__init__(name=name)
        self.config = config

    def __call__(self, batch: feat_batch.Batch) -> Dict[str, jnp.ndarray]:
        """
        Executes the Evoformer trunk over the specified number of recycles.

        Parameters
        ----------
        batch : feat_batch.Batch
            The featurized input batch.

        Returns
        -------
        Dict[str, jnp.ndarray]
            A dictionary containing the 'pair', 'single', and 'target_feat'
            embeddings.
        """
        embedding_module = evoformer_network.Evoformer(
            self.config.evoformer, self.config.global_config
        )
        target_feat = model.create_target_feat_embedding(
            batch=batch,
            config=embedding_module.config,
            global_config=self.config.global_config,
        )

        def recycle_body(
            _, args: Tuple[Dict[str, jnp.ndarray], jnp.ndarray]
        ) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
            prev, key = args
            key, subkey = jax.random.split(key)
            embeddings = embedding_module(
                batch=batch, prev=prev, target_feat=target_feat, key=subkey
            )
            embeddings["pair"] = embeddings["pair"].astype(jnp.float32)
            embeddings["single"] = embeddings["single"].astype(jnp.float32)
            return embeddings, key

        num_res = batch.num_res
        embeddings = {
            "pair": jnp.zeros(
                [num_res, num_res, self.config.evoformer.pair_channel],
                dtype=jnp.float32,
            ),
            "single": jnp.zeros(
                [num_res, self.config.evoformer.seq_channel], dtype=jnp.float32
            ),
            "target_feat": target_feat,
        }

        key = hk.next_rng_key()
        num_iter = self.config.num_recycles + 1
        embeddings, _ = hk.fori_loop(0, num_iter, recycle_body, (embeddings, key))

        # Inject target_feat back into the final dict as the diffusion head expects it
        embeddings["target_feat"] = target_feat
        return embeddings


