"""Compatibility wrapper for the implemented Gold feature entry point."""

from crimenet.jobs.gold_crime_features_job import main

__all__ = ["main"]


if __name__ == "__main__":
    main()
