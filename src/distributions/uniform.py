from distributions.dirichlet import DirichletDistribution
from distributions.dirichlet_multilabel import DirichletMultilabelDistribution


class UniformDistribution:
    def __init__(self, verbose, min_sample_number):
        self.verbose = verbose
        self.min_sample_number = min_sample_number
        self.alpha = 10000  # A large alpha for near-uniform distribution

    def split_to_clients(
        self, df, amount_of_clients, random_state, pathology_names=None
    ):
        is_multilabel = isinstance(df["target"].iloc[0], list)
        if is_multilabel:
            assert (
                pathology_names is not None
            ), "pathology_names must be provided for multilabel classification"
            dirichlet_dist = DirichletMultilabelDistribution(
                alpha=self.alpha,
                verbose=self.verbose,
                min_sample_number=self.min_sample_number,
            )
            return dirichlet_dist.split_to_clients(
                df, amount_of_clients, random_state, pathology_names
            )
        else:
            dirichlet_dist = DirichletDistribution(
                alpha=self.alpha, verbose=self.verbose
            )
            return dirichlet_dist.split_to_clients(df, amount_of_clients, random_state)
