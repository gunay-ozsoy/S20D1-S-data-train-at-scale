import numpy as np
import pandas as pd

from google.cloud import bigquery
from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from taxifare.params import *
from taxifare.ml_logic.data import clean_data
from taxifare.ml_logic.preprocessor import preprocess_features
from taxifare.ml_logic.registry import save_model, save_results, load_model
from taxifare.ml_logic.model import compile_model, initialize_model, train_model
from taxifare.utils import simple_time_and_memory_tracker


def _raw_cache_path(min_date: str, max_date: str) -> Path:
    return Path(LOCAL_DATA_PATH).joinpath(
        "raw", f"query_{min_date}_{max_date}_{DATA_SIZE}.csv"
    )


def _processed_cache_path(min_date: str, max_date: str) -> Path:
    # ✅ tests expect exactly this filename
    return Path(LOCAL_DATA_PATH).joinpath(
        "processed", f"processed_{min_date}_{max_date}_{DATA_SIZE}.csv"
    )


@simple_time_and_memory_tracker
def preprocess(min_date: str = "2009-01-01", max_date: str = "2015-01-01") -> None:
    """
    - Ensure raw query cache exists locally (raw/query_...csv)
    - Stream it by chunks
    - Clean + preprocess each chunk
    - Append to a single processed CSV without headers
    - ✅ Include target (fare_amount) as last column -> 66 columns total
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess" + Style.RESET_ALL)

    min_date = parse(min_date).strftime("%Y-%m-%d")
    max_date = parse(max_date).strftime("%Y-%m-%d")

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM `{GCP_PROJECT_WAGON}`.{BQ_DATASET}.raw_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
    """

    raw_path = _raw_cache_path(min_date, max_date)
    processed_path = _processed_cache_path(min_date, max_date)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Writing processed chunks to: {processed_path}")

    # start fresh each run (avoid appending to old file)
    if processed_path.is_file():
        processed_path.unlink()

    # Create raw cache if missing (tests expect it to exist after preprocess())
    if not raw_path.is_file():
        print("Creating raw cache CSV...")
        client = bigquery.Client(project=GCP_PROJECT)
        data = client.query(query).result().to_dataframe()
        data.to_csv(raw_path, index=False)

    total_written = 0

    for i, chunk in enumerate(
        pd.read_csv(raw_path, parse_dates=["pickup_datetime"], chunksize=CHUNK_SIZE)
    ):
        chunk = clean_data(chunk)

        # X + y
        y_chunk = chunk["fare_amount"].to_numpy().reshape(-1, 1)
        X_chunk = chunk.drop(columns=["fare_amount"])

        X_processed = preprocess_features(X_chunk)  # (n, 65)

        # ✅ concatenate target as last column -> (n, 66)
        data_processed = np.concatenate([X_processed, y_chunk], axis=1)

        # IMPORTANT: tests read with header=None -> write with header=False always
        pd.DataFrame(data_processed).to_csv(
            processed_path,
            mode="a",
            header=False,
            index=False
        )

        total_written += data_processed.shape[0]
        print(f"✅ chunk #{i} saved ({data_processed.shape[0]} rows) | total written: {total_written}")

    print(f"✅ preprocess() done -> {processed_path} ({total_written} rows)")


@simple_time_and_memory_tracker
def preprocess_and_train(min_date: str = "2009-01-01", max_date: str = "2015-01-01") -> None:
    """
    - Query the raw dataset from Le Wagon's BigQuery dataset
    - Cache query result as a local CSV if it doesn't exist locally
    - Clean and preprocess data
    - Train a Keras model on it
    - Save the model
    - Compute & save a validation performance metric
    """
    print(Fore.MAGENTA + "\n ⭐️ Use case: preprocess_and_train" + Style.RESET_ALL)

    min_date = parse(min_date).strftime("%Y-%m-%d")
    max_date = parse(max_date).strftime("%Y-%m-%d")

    query = f"""
        SELECT {",".join(COLUMN_NAMES_RAW)}
        FROM `{GCP_PROJECT_WAGON}`.{BQ_DATASET}.raw_{DATA_SIZE}
        WHERE pickup_datetime BETWEEN '{min_date}' AND '{max_date}'
        ORDER BY pickup_datetime
    """

    data_query_cache_path = _raw_cache_path(min_date, max_date)
    data_query_cached_exists = data_query_cache_path.is_file()

    if data_query_cached_exists:
        print("Loading data from local CSV...")
        data = pd.read_csv(data_query_cache_path, parse_dates=["pickup_datetime"])
    else:
        print("Loading data from Querying Big Query server...")
        data_query_cache_path.parent.mkdir(parents=True, exist_ok=True)

        client = bigquery.Client(project=GCP_PROJECT)
        query_job = client.query(query)
        result = query_job.result()
        data = result.to_dataframe()

        data.to_csv(data_query_cache_path, header=True, index=False)

    data = clean_data(data)

    split_ratio = 0.02
    data = data.sort_values("pickup_datetime").reset_index(drop=True)
    split_index = int(len(data) * (1 - split_ratio))

    data_train = data.iloc[:split_index]
    data_val = data.iloc[split_index:]

    X_train = data_train.drop(columns=["fare_amount"])
    y_train = data_train["fare_amount"].to_numpy()

    X_val = data_val.drop(columns=["fare_amount"])
    y_val = data_val["fare_amount"].to_numpy()

    X_train_processed = preprocess_features(X_train)
    X_val_processed = preprocess_features(X_val)

    learning_rate = 0.0005
    batch_size = 256
    patience = 2

    model = initialize_model(input_shape=(X_train_processed.shape[1],))
    model = compile_model(model, learning_rate=learning_rate)

    model, history = train_model(
        model=model,
        X=X_train_processed,
        y=y_train,
        batch_size=batch_size,
        patience=patience,
        validation_data=(X_val_processed, y_val),
    )

    val_mae = float(np.min(history.history["val_mae"]))

    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience
    )

    save_results(params=params, metrics=dict(mae=val_mae))
    save_model(model=model)

    print("✅ preprocess_and_train() done")


@simple_time_and_memory_tracker
def train(min_date: str = "2009-01-01", max_date: str = "2015-01-01") -> None:
    """
    Incremental training: read processed CSV by chunks and keep training the SAME model.
    Processed CSV format:
      - header=None
      - 66 columns = 65 features + 1 target (fare_amount)
      - dtype float32
    """
    # Read patched values safely at runtime (tests patch taxifare.params.DATA_SIZE/CHUNK_SIZE)
    import taxifare.params as p

    print(Fore.MAGENTA + "\n ⭐️ Use case: train" + Style.RESET_ALL)

    min_date = parse(min_date).strftime("%Y-%m-%d")
    max_date = parse(max_date).strftime("%Y-%m-%d")

    processed_path = Path(p.LOCAL_DATA_PATH).joinpath(
        "processed", f"processed_{min_date}_{max_date}_{p.DATA_SIZE}.csv"
    )

    if not processed_path.is_file():
        raise FileNotFoundError(f"Processed dataset not found: {processed_path}")

    print(f"Loading processed chunks from: {processed_path}")

    n_features = 65
    learning_rate = 0.0005
    batch_size = 256

    # Initialize ONCE
    model = initialize_model(input_shape=(n_features,))
    model = compile_model(model, learning_rate=learning_rate)

    total_seen = 0

    reader = pd.read_csv(
        processed_path,
        header=None,
        dtype=p.DTYPES_PROCESSED,
        chunksize=p.CHUNK_SIZE
    )

    for chunk_id, chunk in enumerate(reader):
        X_chunk = chunk.iloc[:, :n_features].to_numpy()
        y_chunk = chunk.iloc[:, n_features].to_numpy()

        model.fit(
            X_chunk,
            y_chunk,
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )

        total_seen += len(chunk)
        print(f"✅ trained on chunk #{chunk_id} ({len(chunk)} rows) | total seen: {total_seen}")

    save_model(model)
    print("✅ train() done")


def pred(X_pred: pd.DataFrame = None) -> np.ndarray:
    print(Fore.MAGENTA + "\n ⭐️ Use case: pred" + Style.RESET_ALL)

    if X_pred is None:
        X_pred = pd.DataFrame(
            dict(
                pickup_datetime=[pd.Timestamp("2013-07-06 17:18:00", tz="UTC")],
                pickup_longitude=[-73.950655],
                pickup_latitude=[40.783282],
                dropoff_longitude=[-73.984365],
                dropoff_latitude=[40.769802],
                passenger_count=[1],
            )
        )

    model = load_model()
    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print("✅ pred() done")
    return y_pred


if __name__ == "__main__":
    try:
        # preprocess()
        preprocess_and_train()
        # train()
        pred()
    except Exception:
        import sys
        import traceback
        import ipdb

        extype, value, tb = sys.exc_info()
        traceback.print_exc()
        ipdb.post_mortem(tb)
