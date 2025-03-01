import tempfile
import os
import aiofiles
import pillow_heif
from PIL import Image, ExifTags
import logging

logger = logging.getLogger(__name__)

async def save_upload_file(upload_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upload_file.filename)[1]) as temp_file:
            file_path = temp_file.name
            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await upload_file.read()
                await out_file.write(content)
        return file_path
    except Exception as e:
        logger.error(f"Could not save file: {str(e)}")
        raise

async def convert_heif_to_jpeg(file_path):
    try:
        if not pillow_heif.is_supported(file_path):
            raise ValueError(f"File is not a supported HEIF image: {file_path}")

        heif_file = pillow_heif.read_heif(file_path)
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        
        jpeg_path = file_path.rsplit('.', 1)[0] + '.jpg'
        image.save(jpeg_path, format="JPEG")
        
        logger.info(f"Successfully converted {file_path} to {jpeg_path}")
        return jpeg_path
    except Exception as e:
        logger.error(f"Error converting HEIF to JPEG: {e}")
        raise

def fix_image_rotation(image_path):
    try:
        image = Image.open(image_path)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())
        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
        image.save(image_path)
        image.close()
        logger.info(f"Image rotation fixed for {image_path}")
    except (AttributeError, KeyError, IndexError):
        logger.info(f"No Exif orientation found for {image_path}")
    except Exception as e:
        logger.error(f"Error fixing image rotation: {str(e)}")