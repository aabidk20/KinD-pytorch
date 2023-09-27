import argparse


class BaseParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument("--mode",default="train", choices=["train", "test"])
        self.parser.add_argument("--config", default="config.yaml", help="path to config file")
        self.parser.add_argument("-c", "--checkpoint", default="./weights/",
                                 help="Path of checkpoints")
        # BUG: why is default for checkpoint True? is it a flag?
        return self.parser.parse_args()