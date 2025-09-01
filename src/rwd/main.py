from src.rwd.color_extraction import process_image_input
from src.rwd.axes_detection import detect_vertical_axes
from src.rwd.cropping import crop_images
from src.rwd.category_separation import process_images_separation

def main():
    input_path = "real_world_test_data/final_set_raw_images/done"
    output_dir = "real_world_test_data/final_vertical_axes/images"
    dominant_colors = "dominant_colors.json"
    ver_ann = "vertical_annotations.json"
    cropped = "cropped_output"
    sep_output = "separated_output"
    sep_json = "separated_output/colors.json"

    process_image_input(input_path, output_json=dominant_colors)
    detect_vertical_axes(input_path, output_dir, ver_ann)
    crop_images(output_dir, ver_ann, cropped)
    process_images_separation(cropped, sep_output, sep_json, use_kmeans=True)

if __name__ == "__main__":
    main()
