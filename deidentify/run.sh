python /home/eo287/github/ccta-pipeline/deidentify/deidentify-ccta.py \
  --input /home/eo287/mnt/s3_ccta/cta_09232025/studies/E100138698/1.2.840.113845.11.1000000001799338748.20130201075411.7339398 \
  --eid E100138698 \
  --out-root /home/eo287/mnt/s3_ccta/deidentified \
  --log-root /home/eo287/mnt/s3_ccta/deidentified-logs \
  --min-slices 16 \
  --workers 8
