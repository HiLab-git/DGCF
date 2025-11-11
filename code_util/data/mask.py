
import numpy as np

def segMask2binaryMasks(multiClassMask, class_range = None, fuse=False):
    """
    Convert a multi-class mask to binary masks for each class.
    
    Parameters:
    - mask: numpy array of shape (H, W, D) or (H, W), representing the multi-class mask.
    - fuse: boolean, if True, append a binary where all the non-zero classes are fused into one.
    
    Returns:
    - binary_masks: list of numpy arrays, each representing a binary mask for a class.
    """

    # Check if the input is a 3D or 2D array
    if len(multiClassMask.shape) == 3:
        H, W, D = multiClassMask.shape
    elif len(multiClassMask.shape) == 2:
        H, W = multiClassMask.shape
        D = 1
        multiClassMask = np.expand_dims(multiClassMask, axis=-1)
    else:
        raise ValueError("Input mask must be either 2D or 3D.")

    # check if the mask is multi-class or binary
    if np.max(multiClassMask) <= 1:
        # If already binary, just use it as is
        binary_masks = [multiClassMask.astype(np.uint8)]
        return binary_masks
        
    # Get unique classes
    if class_range is not None:
        # Filter unique classes based on the provided class range
        unique_classes = range(1, len(class_range) + 1)  # Assuming class_range is a list of tuples defining ranges for classes
    else:
        unique_classes = np.unique(multiClassMask)
    # Create binary masks for each class
    binary_masks = []
    unique_classes_ref = np.unique(multiClassMask)
    for cls in unique_classes:
        if cls in unique_classes_ref:
            binary_mask = (multiClassMask == cls).astype(np.uint8)
            binary_masks.append(binary_mask)
        else:
            binary_masks.append(None)

    # Optionally fuse all classes into one binary mask
    if fuse:
        binary_masks_temp = [mask for mask in binary_masks if mask is not None]  # Remove None masks
        fused_mask = np.sum(binary_masks_temp, axis=0)
        binary_masks.append(fused_mask.astype(np.uint8))

    return binary_masks

def generateSegMask(image,class_range,nonzero = True):
    """
    generate segmentation mask with a reference image and class range
    values outside the range are set to 0, values inside the range are set to 1 to n
    Parameters:
    - image: numpy array of shape (H, W, D) or (H, W), representing the reference image.
    - class_range: list of tuples, each tuple contains the range of values for a class.
    """
    # Check if the input is a 3D or 2D array
    if len(image.shape) == 3:
        H, W, D = image.shape
    elif len(image.shape) == 2:
        H, W = image.shape
        D = 1
        image = np.expand_dims(image, axis=-1)
    else:
        raise ValueError("Input mask must be either 2D or 3D.")

    # Initialize the segmentation mask
    seg_mask = np.zeros((H, W, D), dtype=np.uint8)

    # Generate the segmentation mask based on the class ranges
    for i, (start, end) in enumerate(class_range):
        seg_mask[(image >= start) & (image <= end)] = i + 1
    # Set values outside the range to the closest class
    if nonzero: 
        seg_mask[image < class_range[0][0]] = 1
        seg_mask[image > class_range[-1][1]] = len(class_range)
    return seg_mask