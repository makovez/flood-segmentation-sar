import argparse
from flood_mapping.evaluate import ModelEvaluate

def main(data_path, model_path, patch_size, save_images):
    evaluator = ModelEvaluate(model_path=model_path, data_folder=data_path, patch_size=patch_size)
    tiles = evaluator.evaluate()
    if save_images:
        evaluator.save_images(tiles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Flood Mapping Model")
    parser.add_argument("--data_path", type=str, required=True, help="Path to evaluation data")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--patch_size", type=int, default=128, help="Patch size of images")
    parser.add_argument("--save_images", action="store_true", help="Save output images (default: False)")
    args = parser.parse_args()

    main(args.data_path, args.model_path, args.patch_size, args.save_images)

# examples
    # python evaluate.py --data_path data --model_path model/unet-voc.pt --patch_size 128 --save_images