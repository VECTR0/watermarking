import argparse
import os

from PIL import Image


def count_images_and_size(folder_path: str) -> None:
    image_extensions = {".jpg": 0, ".jpeg": 0, ".png": 0}
    image_sizes = []
    resolutions = []
    total_size = 0

    for root, dirs, files in os.walk(folder_path):
        for file_ in files:
            file_lower = file_.lower()
            file_path = os.path.join(root, file_)

            for ext in image_extensions:
                if file_lower.endswith(ext):
                    image_extensions[ext] += 1
                    file_size = os.path.getsize(file_path)
                    image_sizes.append(file_size)
                    total_size += file_size

                    try:
                        with Image.open(file_path) as img:
                            width, height = img.size
                            resolutions.append((width, height))
                    except Exception as e:
                        print(f"Warning: Could not open image {file_path}: {e}")
                    break

    total_count = sum(image_extensions.values())
    avg_size = total_size / total_count if total_count > 0 else 0
    largest_size = max(image_sizes) if image_sizes else 0
    smallest_size = min(image_sizes) if image_sizes else 0

    largest_resolution = max(resolutions, key=lambda x: x[0] * x[1], default=(0, 0))
    smallest_resolution = min(resolutions, key=lambda x: x[0] * x[1], default=(0, 0))
    avg_resolution = (
        sum(width for width, height in resolutions) // len(resolutions)
        if resolutions
        else 0,
        sum(height for width, height in resolutions) // len(resolutions)
        if resolutions
        else 0,
    )

    return {
        "format_counts": image_extensions,
        "total_count": total_count,
        "total_size": total_size,
        "avg_size": avg_size,
        "largest_size": largest_size,
        "smallest_size": smallest_size,
        "largest_resolution": largest_resolution,
        "smallest_resolution": smallest_resolution,
        "avg_resolution": avg_resolution,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Count images by format, calculate folder size, and gather resolution info."
    )
    parser.add_argument("folder_path", type=str, help="Path to the folder")
    args = parser.parse_args()

    result = count_images_and_size(args.folder_path)

    print("Image Counts by Format:")
    for ext, count in result["format_counts"].items():
        print(f"{ext.upper()}: {count}")

    print(f"\nTotal Image Count: {result['total_count']}")
    print(f"Total Size: {result['total_size'] / (1024 ** 2):.2f} MB")
    print(f"Average Image Size: {result['avg_size'] / 1024:.2f} KB")
    print(f"Largest Image Size: {result['largest_size'] / 1024:.2f} KB")
    print(f"Smallest Image Size: {result['smallest_size'] / 1024:.2f} KB")

    print("\nResolution Information:")
    print(
        f"Average Resolution: {result['avg_resolution'][0]}x{result['avg_resolution'][1]}"
    )
    print(
        f"Largest Resolution: {result['largest_resolution'][0]}x{result['largest_resolution'][1]}"
    )
    print(
        f"Smallest Resolution: {result['smallest_resolution'][0]}x{result['smallest_resolution'][1]}"
    )


if __name__ == "__main__":
    main()
