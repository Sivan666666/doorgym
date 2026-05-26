#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .configuration_pi05_evo import PI05EvoConfig
# processor eager-imported for @ProcessorStepRegistry.register side effects.
from .processor_pi05_evo import make_pi05_evo_pre_post_processors


def __getattr__(name):
    # Lazy-load modeling only — it pulls transformers + torchvision (~25s).
    if name == "PI05EvoPolicy":
        from .modeling_pi05_evo import PI05EvoPolicy
        return PI05EvoPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["PI05EvoConfig", "PI05EvoPolicy", "make_pi05_evo_pre_post_processors"]
