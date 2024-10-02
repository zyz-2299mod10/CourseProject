import splitfolders


if __name__ == "__main__":
    splitfolders.ratio("./data_bird/train", output="./data-split", seed=1337, ratio=(0.7, 0.3), group_prefix=None, move=False)
