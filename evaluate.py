from flood_mapping.evaluate import ModelEvaluate

model_evaluate = ModelEvaluate()
tiles = model_evaluate.evaluate()
model_evaluate.save_images(tiles)