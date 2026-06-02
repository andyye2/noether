#  Copyright © 2025 Emmi AI GmbH. All rights reserved.


from noether.core.schemas.models import TransolverConfig
from noether.modeling.models import Transformer


class Transolver(Transformer):
    """Implementation of the Transolver model.
    Reference code: https://github.com/thuml/Transolver/
    Paper: https://arxiv.org/abs/2402.02366
    Transolver is a Transformer with a special physics attention mechanism. Hence, we extend the Transformer class,
    and configure it accordingly.
    """

    def __init__(
        self,
        config: TransolverConfig,
    ):
        """

        Args:
            config: Configuration of the Transolver model.
        """

        super().__init__(config=config)
