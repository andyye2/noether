#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import uuid


class StopForwardException(Exception):
    pass


class ForwardHook:
    def __init__(self, outputs: dict, raise_exception: bool = False, key: str | None = None):
        self.key = key or uuid.uuid4()
        self.outputs = outputs
        self.raise_exception = raise_exception
        self.enabled = True

    def __call__(self, _, __, output):
        if not self.enabled:
            return
        self.outputs[self.key] = output
        if self.raise_exception:
            raise StopForwardException
