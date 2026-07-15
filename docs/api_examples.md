# quick API examples

single prediction:
  curl -X POST localhost:8501/api/predict \
    -H "Content-Type: application/json" \
    -d '{"age": 52, "bp": 128}'

batch upload:
  curl -X POST localhost:8501/api/predict/batch \
    -F "file=@./batch.csv"

history: GET /api/history?limit=20
