web: dvc config core.hardlink_lock true && dvc config core.no_scm true && dvc pull -v &&cd starter && AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY AWS_REGION=us-east-1 uvicorn app.api:app --host 0.0.0.0 --port $PORT
