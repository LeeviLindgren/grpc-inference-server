use image::{ImageBuffer, Luma};

fn main() {
    // Read the pic path from sys args
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <image_path>", args[0]);
        return;
    }
    let img_path = &args[1];
    println!("Reading image from {}", img_path);

    // Load the image from a file
    let img = image::open(img_path).unwrap();

    // or, if you have the image data in memory:
    // let img = image::load_from_memory(image_data).unwrap();

    // Example of converting to a grayscale image
    let gray_img: ImageBuffer<Luma<u8>, Vec<u8>> = img.to_luma8();

    // Resize the image to 28x28 pixels
    let resized_img =
        image::imageops::resize(&gray_img, 28, 28, image::imageops::FilterType::Triangle);

    // Example of saving the grayscale image
    resized_img.save("out.png").unwrap();
}
