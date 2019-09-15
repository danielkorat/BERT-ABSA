
import argparse
import abc


class Procedure:
    def __init__(self, parser=None):
        if parser is None:
            parser = argparse.ArgumentParser()
        self.parser = parser

    @staticmethod
    @abc.abstractmethod
    def add_arguments(parser: argparse.ArgumentParser):
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def run_procedure(args):
        raise NotImplementedError

    @classmethod
    def run(cls):
        parser = argparse.ArgumentParser()
        cls.add_arguments(parser)
        cls.run_procedure(parser.parse_args())
