from types import MethodType


class BaseSelector:
    def __init__(self, cfg=None):
        self.cfg = cfg

    # Main function to setup
    # client sampling strategy run-time
    def __call__(self, trainer):
        # Setup all neccessary stuff for method
        self.setup_strategy(trainer)

        # Replace trainer methods
        # on functions for a specific client selection method
        trainer = self.change_functionality(trainer)

        # Don`t forget return updated trainer
        return trainer

    def setup_strategy(self, trainer):
        # Here is the logic of setting up the client selection method itself
        # Setting up parameters and variables
        pass

    def change_functionality(self, trainer):
        # Use MethodType for rewrite methods of server class
        # Like this:
        # trainer.server.select_clients_to_train = MethodType(
        #     BaseSelector.select_clients_to_train, trainer.server
        # )
        return trainer
