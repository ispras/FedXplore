class BasePreaggregator:
    def __init__(self, server):
        "We pass server as an attribute to access the model state and cfg."
        self.server = server

    def pre_aggregate(self, client_gradients):
        raise NotImplementedError(
            "This method must be implemented in custom preaggregator"
        )
