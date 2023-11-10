# For a given svs file, sequentially non-overlapping 256x256 images are extracted and placed into the HR folder.
# https://openslide.org/api/python/#installing
# Download the binary file from the openslide official website, extract it, add the 'bin' path to the system PATH,
# and then paste it into the OPENSLIDE_PATH below. Don't forget to install openslide-python with 'pip install openslide-python'.

OPENSLIDE_PATH = r'E:\Python_study\openslide-win64-20231011\bin'

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide

# Set the path for the input .svs file
svs_file_path = 'E:\Python_study\HTMCP-03-06-02412-01-HE.8D394225-54F0-45DC-ABC1-85256244EA6F.svs'

# Save to the HR_svs folder
output_dir = 'HR_svs'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the .svs file
slide = openslide.OpenSlide(svs_file_path)

# Define the size of image tiles to extract
tile_width = 256
tile_height = 256

# Get the width and height of the slide
slide_width = slide.dimensions[0]
slide_height = slide.dimensions[1]
print(f'slide_width: {slide_width}, slide_height: {slide_height}')

# Loop through the slide and extract image tiles
tile_count = 0
for x in range(0, slide_width, tile_width):
    for y in range(0, slide_height, tile_height):
        # Get the image tile
        tile = slide.read_region((x, y), 0, (tile_width, tile_height))

        # Convert the image tile to RGB format
        tile_rgb = tile.convert('RGB')

        # Modify the naming format of the image and save the image tile
        tile_output_path = os.path.join(output_dir, f'tile_{tile_count}.png')
        tile_rgb.save(tile_output_path)

        tile_count += 1

print(f"Total tiles saved: {tile_count}")
