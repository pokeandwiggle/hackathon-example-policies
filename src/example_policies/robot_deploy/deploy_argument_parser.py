import argparse
from pathlib import Path


class DeployArgumentParser:
    """Shared argument parser for deployment scripts."""

    @staticmethod
    def add_common_args(parser: argparse.ArgumentParser):
        """Add arguments common to all deploy scripts."""
        parser.add_argument(
            "-s",
            "--robot-server",
            default="localhost:50051",
            metavar="ADDR",
            help="Robot service server address (default: localhost:50051)",
        )
        parser.add_argument(
            "-z",
            "--hertz",
            type=float,
            default=10.0,
            metavar="HZ",
            help="Control frequency in Hz (default: 10.0)",
        )

    @staticmethod
    def create_single_policy_parser() -> argparse.ArgumentParser:
        """Create parser for single-policy deployment."""
        parser = argparse.ArgumentParser(description="Single policy deployment")
        parser.add_argument(
            "-c",
            "--checkpoint",
            type=Path,
            required=True,
            metavar="PATH",
            help="Path to the policy checkpoint directory",
        )
        DeployArgumentParser.add_common_args(parser)
        return parser

    @staticmethod
    def create_multi_policy_parser() -> argparse.ArgumentParser:
        """Create parser for multi-policy deployment."""
        parser = argparse.ArgumentParser(
            description="Multi-policy deployment with chaining"
        )
        parser.add_argument(
            "-c",
            "--checkpoints",
            type=Path,
            required=True,
            nargs="+",
            metavar="PATH",
            help="Paths to the policy checkpoint directories (space-separated)",
        )
        parser.add_argument(
            "-p",
            "--prompt-interval",
            type=int,
            default=50,
            metavar="N",
            help="Prompt for policy selection every N steps (default: 50, 0 to disable)",
        )
        DeployArgumentParser.add_common_args(parser)
        return parser
