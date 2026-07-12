# input formats

## single record
flat json of feature_name: value, one record per request.
unknown keys are ignored, missing keys fail validation with a 422.

## batch
the batch endpoint takes a csv (utf-8, comma separated) with a header
row matching the feature names. row-level errors are reported per line
instead of failing the whole upload.

## dates
date columns should arrive as ISO strings (YYYY-MM-DD), the pipeline
converts them internally.
