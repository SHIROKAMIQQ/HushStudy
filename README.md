## Installing Dependencies

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Dataset Management

You may use `automated_extraction.py` to extract features out of .wav files into .csv files. 

Be sure to put the .csv file into either `chatter_classifier_datasets/` or `duration_prediction_datasets/`, depending on which dataset it is for.

For .csv files in `duration_prediction_datasets/` you might want to run `chatter_classifier_post_processing.py` on them to get extra features needed for duration prediction.

Then, run `combine_datasets.py` to get a `master.csv` file in both dataset folders. This `master.csv` file will be the one used by the respective models.

## Hosting the server

To host the server (assuming localhost), run:
```
uvicorn main:app --host 127.0.0.1 --port 8000
```

Now, your server is accessible via `http://localhost:8000/`.


