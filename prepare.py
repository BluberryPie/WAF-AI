import conf.preprocess as cfg
import logging
import pandas as pd

from src.preprocess import remove_separators, remove_url_encoding


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
    try:
        train_df["payload"] = train_df["payload"].apply(remove_separators)
        train_df["payload"] = train_df["payload"].apply(remove_url_encoding)
    except Exception as e:
        logging.error(f"{e}\nFailed to preprocess payloads")
    else:
        logging.info("[*] Successfully preprocessed all payloads")

    # 3. Train the BPE tokenizer

    # 4. Train the Word2Vec model


if __name__ == "__main__":

    logging.basicConfig(format="%(message)s", level=logging.INFO)
    main()

