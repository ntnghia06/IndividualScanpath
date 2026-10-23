from bootstrap import BRANCH
from gazeformer_transfer.engine import train_main

if __name__ == "__main__":
    train_main("embedding", BRANCH)
