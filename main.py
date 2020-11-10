import os

from PIL import Image
import numpy as np

from style_transfer import perform_style_transfer

def main():
    img_dir = "docs"
    content_image_path = os.path.join(img_dir, "content_img.jpg")
    style_image_path = os.path.join(img_dir, "style_img.jpg")

    content_image = np.asarray(Image.open(content_image_path))
    style_image = np.asarray(Image.open(style_image_path))

    hyper_parameters = {
        "content_style_ratio_weight": 0.0001,
        "num_iterations": 1000
    }

    output_image = perform_style_transfer(content_image, style_image, hyper_parameters)

    output_image_path = os.path.join(img_dir, "output_img_{}_{}.png".format(
        hyper_parameters["num_iterations"], hyper_parameters["content_style_ratio_weight"]))
    Image.fromarray(output_image).save(output_image_path)

if __name__ == "__main__":
    main()