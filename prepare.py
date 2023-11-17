import conf.preprocess as cfg
import logging
import pandas as pd


def main():

    # 1. Read the training data
    try:
        train_df = pd.read_csv(cfg.TRAIN_DATA_FILEPATH)
    except Exception as e:
        logging.error(f"{e}\nFailed to read the training data")
    else:
        logging.info("[*] Successfully read the training data")
        logging.info(f"[*] Number of rows: {train_df.shape[0]}")

    # 2. Preprocess payloads

    # 3. Train the BPE tokenizer

    # 4. Train the Word2Vec model


if __name__ == "__main__":

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main()

