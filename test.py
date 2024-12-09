import mmcv

# Load the predictions file
predictions = mmcv.load('/home/klingjac/UniAD/output/results.pkl')

# Access 'bbox_results'
bbox_results = predictions.get('bbox_results', None)
if bbox_results and isinstance(bbox_results, list):
    print(f"First entry in 'bbox_results': {bbox_results[0]}")
else:
    print("Unexpected structure for 'bbox_results'.")
