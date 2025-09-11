# Copyright 2025 Poke & Wiggle GmbH. All rights reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import dataclasses
from enum import Enum
from typing import Type

from .pipeline_config import PipelineConfig


def add_dataclass_to_argparser(parser: argparse.ArgumentParser, dataclass_type: Type):
    for field in dataclasses.fields(dataclass_type):
        arg_name = "--" + field.name.replace("_", "-")
        default = field.default
        field_type = field.type

        # Handle Enum fields
        if isinstance(default, Enum) or (
            isinstance(field_type, type) and issubclass(field_type, Enum)
        ):
            parser.add_argument(
                arg_name,
                type=str,
                choices=[e.value for e in field_type],
                default=default.value if isinstance(default, Enum) else default,
                help=f"{field.name} (default: {default})",
            )
        # Handle bool as store_true/store_false
        elif field_type is bool:
            parser.add_argument(
                arg_name,
                action="store_true" if default is False else "store_false",
                default=default,
                help=f"{field.name} (default: {default})",
            )
        # Handle tuples (e.g., image_resolution)
        elif field_type is tuple:
            parser.add_argument(
                arg_name,
                type=int,
                nargs=2,
                default=default,
                help=f"{field.name} (default: {default})",
            )
        else:
            parser.add_argument(
                arg_name,
                type=field_type,
                default=default,
                help=f"{field.name} (default: {default})",
            )


def parse_pipeline_config_from_args(parser: argparse.ArgumentParser):
    existing_arg_names = set()
    for action in parser._actions:
        if action.dest != "help":  # Skip the default help argument
            existing_arg_names.add(action.dest)

    add_dataclass_to_argparser(parser, PipelineConfig)
    args = parser.parse_args()
    kwargs = vars(args)

    existing_args = {}
    pipeline_kwargs = {}

    for key, value in kwargs.items():
        if key in existing_arg_names:
            existing_args[key] = value
        else:
            pipeline_kwargs[key] = value

    # Convert enum fields back to Enum instances (only for PipelineConfig fields)
    for field in dataclasses.fields(PipelineConfig):
        if field.name in pipeline_kwargs and isinstance(field.default, Enum):
            enum_type = type(field.default)
            pipeline_kwargs[field.name] = enum_type(pipeline_kwargs[field.name])

    # Create a new args namespace with both existing and pipeline args
    combined_args = argparse.Namespace(**{**existing_args, **pipeline_kwargs})
    return PipelineConfig(**pipeline_kwargs), combined_args
