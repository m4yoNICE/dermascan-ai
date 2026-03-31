from PIL import Image, UnidentifiedImageError
from io import BytesIO
import numpy as np
from typing import Union, Tuple

# Configuration
TARGET_SIZE = (448, 448) 
TARGET_SIZE_DERM = (448, 448)  
TARGET_SIZE_EFFICIENTNET = (224, 224)  

MAX_IMAGE_SIZE_MB = 10
MAX_IMAGE_BYTES = MAX_IMAGE_SIZE_MB * 1024 * 1024


class ImagePreprocessingError(Exception):
    pass


def validate_image_data(image_data: bytes):

    if len(image_data) > MAX_IMAGE_BYTES:
        raise ImagePreprocessingError(
            f"Image exceeds {MAX_IMAGE_SIZE_MB}MB limit "
            f"({len(image_data) / (1024**2):.1f}MB)"
        )
    
    if len(image_data) == 0:
        raise ImagePreprocessingError("Empty image data")


def load_image(image_data: Union[bytes, str]):
    try:
        if isinstance(image_data, bytes):
            # Validate size first
            validate_image_data(image_data)
            
            # Load from bytes
            img = Image.open(BytesIO(image_data))
        else:
            # Load from file path
            img = Image.open(image_data)
        
        # Verify it's a valid image (doesn't fully load into memory)
        img.verify()
        
        # Reopen after verify (verify closes the file)
        if isinstance(image_data, bytes):
            img = Image.open(BytesIO(image_data))
        else:
            img = Image.open(image_data)
            
        return img
        
    except UnidentifiedImageError:
        raise ImagePreprocessingError("Invalid or corrupted image format")
    except FileNotFoundError:
        raise ImagePreprocessingError(f"Image file not found: {image_data}")
    except Exception as e:
        raise ImagePreprocessingError(f"Failed to load image: {str(e)}")


def preprocess_for_derm_foundation(image_data: Union[bytes, str], target_size: Tuple[int, int] = TARGET_SIZE_DERM
):
    try:
        # Load image
        img = load_image(image_data)
        
        # Convert to RGB (handles RGBA, grayscale, etc.)
        img = img.convert("RGB")
        
        # Resize
        img = img.resize(target_size, Image.LANCZOS)
        
        return img
        
    except ImagePreprocessingError:
        raise
    except Exception as e:
        raise ImagePreprocessingError(f"Preprocessing failed: {str(e)}")


def preprocess_for_efficientnet(image_data: Union[bytes, str], target_size: Tuple[int, int] = TARGET_SIZE_EFFICIENTNET):
    try:
        img = load_image(image_data)
        img = img.convert("RGB")
        img = img.resize(target_size, Image.LANCZOS)
        return img
        
    except ImagePreprocessingError:
        raise
    except Exception as e:
        raise ImagePreprocessingError(f"Preprocessing failed: {str(e)}")
    
def image_to_array(img: Image.Image, normalize: bool = False):
    img_array = np.array(img, dtype=np.float32)
    
    if normalize:
        img_array = img_array / 255.0
    
    return img_array


def preprocess_to_array(
    image_data: Union[bytes, str],
    normalize: bool = False,
    add_batch_dim: bool = False
):

    img = preprocess_for_derm_foundation(image_data)
    img_array = image_to_array(img, normalize=normalize)
    
    if add_batch_dim:
        img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def preprocess_to_array_efficientnet(image_data: Union[bytes, str], add_batch_dim: bool = True):
    try:
        img = load_image(image_data)
        img = img.convert("RGB")
        img = img.resize(TARGET_SIZE_EFFICIENTNET, Image.LANCZOS)
        
        img_array = np.array(img, dtype=np.float32)
        
        if add_batch_dim:
            img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except ImagePreprocessingError:
        raise
    except Exception as e:
        raise ImagePreprocessingError(f"EfficientNet preprocessing failed: {str(e)}")


def image_to_png_bytes(img: Image.Image):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# Convenience function for backward compatibility
def preprocess_image(image_data: Union[bytes, str]) -> Image.Image:
    return preprocess_for_derm_foundation(image_data)