import pydicom as dicom
import numpy as np
import matplotlib.pyplot as plt
import os
import skimage.transform
from skimage import exposure
from scipy.ndimage import zoom
import cv2
OPENSLIDE_PATH = "C:\\Users\\peter\\Documents\\Uni\\Second_Year\\MDP\\Openslide\\openslide-bin-4.0.0.3-windows-x64\\bin"
with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
import openslide
from openslide import open_slide

from skimage.filters import threshold_otsu
import math


dcm_ext = '.dcm'

def get_volume_clinicalTrialTimePoint(folder_path):
    '''
        This function returns a dictionary (I guess) with the metadata of the dicom volume in the folder given in input
        Args:
            folder_path (str): path to the folder containing the volume in dicom files
        Returns:
            a dictionary 

    '''
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path,f))]
    dcm_image = dicom.dcmread(os.path.join(folder_path,files[0]))
    return dcm_image.ClinicalTrialTimePointID

def load_single_volume(folder_path):
    '''
        This function returns a tuple with the volume of a single patient, its mean voxel dimension,
        and a list of DICOM slices metadata (including z-coordinates).
        
        Args:
            folder_path (str): path to the folder containing the volume in dicom files
        Returns:
            a tuple ( the volume as np.array, the mean voxel dimension of the volume, list of DICOM slices metadata)
    '''
    img_vol = []
    voxel_z = []
    voxel_x = []
    voxel_y = []
    
    dicom_slices = []  # List to store DICOM slices metadata
    
    spacing = None
    
    # Handle long paths
    folder_path = os.path.abspath(folder_path)

    if folder_path.startswith(u"\\\\"):
        folder_path = u"\\\\?\\UNC\\" + folder_path[2:]
    else:
        folder_path = u"\\\\?\\" + folder_path
        
    dcm_ext = ".dcm"  # Assuming DICOM files have the extension '.dcm'
        
    for path, _, files in sorted(os.walk(folder_path)): 
        for filename in sorted(files): 
            if filename.endswith(dcm_ext):
                img_dcm_std = dicom.dcmread(os.path.join(folder_path, filename))
                img = img_dcm_std.pixel_array
                img_vol.append(img)
                
                if not hasattr(img_dcm_std, 'SpacingBetweenSlices'): 
                    spacing = 0
                else:
                    spacing = img_dcm_std.SpacingBetweenSlices
                    
                voxel_z.append(spacing)
                voxel_x.append(img_dcm_std.PixelSpacing[0])
                voxel_y.append(img_dcm_std.PixelSpacing[1])
                
                # Append the DICOM slice to the list (to access metadata like ImagePositionPatient)
                dicom_slices.append(img_dcm_std)
      
    voxel_z = np.array(voxel_z)
    voxel_x = np.array(voxel_x)
    voxel_y = np.array(voxel_y)      
    z_space = voxel_z.mean()
    x_space = voxel_x.mean()
    y_space = voxel_y.mean()
    vox_dim = (x_space, y_space, z_space)
    
    return np.array(img_vol), vox_dim, dicom_slices

def sget_occupied_slices(rtstruct_path, dicom_slices):
    # Load the RTSTRUCT file
    rtstruct = dicom.read_file(rtstruct_path)
    
    # Extract the Contour Sequences
    contour_sequences = rtstruct.ROIContourSequence
    
    # Get the z-coordinates of the DICOM slices
    slice_z_positions = [float(slice.ImagePositionPatient[2]) for slice in dicom_slices]
    #if rtstruct.PatientID.strip() == "C3L-02118":
    #            print("Slice z positions: \n",slice_z_positions)
    # Initialize a set to store occupied slice indices
    occupied_slices = set()
    #if rtstruct.PatientID.strip() == "C3L-02118":
    #    print()
    # Loop through each contour sequence
    for contour_seq in contour_sequences:
        for contour in contour_seq.ContourSequence:
            # Extract contour data (x, y, z coordinates)
            contour_data = contour.ContourData
            
            # Extract z-values from contour data
            contour_z_values = contour_data[2::3]  # Every third value is a z-coordinate
            #if rtstruct.PatientID.strip() == "C3L-02118":
            #    print("contour values of C3L-02118: \n",contour_z_values) 
                
            # Match contour z-values with slice z-positions
            for z in contour_z_values:
            # Iterate over the slice_z_positions to check for close values
                for i, slice_z in enumerate(slice_z_positions):
                    # Check if z is close to slice_z within the threshold
                    if math.isclose(z, slice_z, abs_tol=0.9):
                        slice_index = i  # Get the index of the matching z position
                        occupied_slices.add(slice_index)  # Add the index to the occupied_slices set
    
    return sorted(list(occupied_slices))

def remap_occupied_slices(occupied_slices, zoom_factor_z):
    '''
    Remap the indices of occupied slices from the original volume to the normalized volume.
    
    Args:
        occupied_slices (list): List of occupied slice indices in the original volume.
        zoom_factor_z (float): The zoom factor applied to the z-axis (slice dimension).
    
    Returns:
        list: Remapped slice indices in the normalized volume.
    '''
    '''
    Remap and fill the indices of occupied slices from the original volume to the normalized volume.
    
    Args:
        occupied_slices (list): List of occupied slice indices in the original volume.
        zoom_factor_z (float): The zoom factor applied to the z-axis (slice dimension).
    
    Returns:
        list: Remapped and filled slice indices in the normalized volume.
    '''
    remapped_slices = []

    # Step 1: Remap each slice index using the zoom factor
    for i in range(len(occupied_slices)):
        # Remap current slice
        current_mapped = int(round(occupied_slices[i] * zoom_factor_z))
        
        if i > 0:
            # Remap previous slice
            previous_mapped = int(round(occupied_slices[i - 1] * zoom_factor_z))
            
            # If the current and previous slices are subsequent in the original,
            # Fill in the indices between them
            if occupied_slices[i] == occupied_slices[i - 1] + 1:
                remapped_slices.extend(range(previous_mapped, current_mapped + 1))
            else:
                remapped_slices.append(current_mapped)
        else:
            remapped_slices.append(current_mapped)
    
    return sorted(set(remapped_slices))  # Remove duplicates and sort

def preprocess(volume : np.array, target_shape : list) -> np.array:
    ''' This function normalizes the volume and returns it in the given target shape

        Args:
            volume (np.array): the volume to be normalized (n_slices,slice_height,slice_width)
            target_shape (list): list containing the target shape of the volume [number of slices, height, width]
        
        Returns:
            The normalized volume as a numpy array 
            
    '''
    factor = (
            target_shape[0]/volume.shape[0],
            target_shape[1]/volume.shape[1],
            target_shape[2]/volume.shape[2]
        )
    volume = zoom(volume, factor, order = 3, mode = 'nearest')
    #max min normalization
    m = np.mean(volume)
    s = np.std(volume)
    return np.divide((volume - m), s), factor  

#--------------------------------------WSI Preprocessing Functions-----------------------
wsi_ext = '.svs'
def get_patients_ids(dataset_path):
    '''
    This function returns the set of patients ids in the wsi dataset folder. 
    This function assumes that it's a unique folder with svs files named like: C3L-00189-21.svs 
    Args:
        dataset_path (str)
    Returns:
        The set of unique patients ids in the dataset 
    '''
    patients_ids = []
    for path, _, files in sorted(os.walk(dataset_path)): 
        for filename in sorted(files):
          if filename.endswith(wsi_ext):
            patient_id = filename.split('.')[0]
            patient_id = patient_id[:-3]
            patients_ids.append(patient_id)
    patients_ids = set(patients_ids)
    return patients_ids

def read_wsi(path):
    ''' This function uses the openslide library to open a wsi image from path and return it
        Args:
            path (str): path to the whole slide image
        Returns: An OpenSlide object containing the wsi and its metadata informations
    '''
    slide = openslide.OpenSlide(path)
    return slide

def get_thumbnail(slide, level):
    ''' This function retrieves, through the openslide library, the image at the magnification level provided
        Args:
            slide (OpenSlide object): the slide from which to retrieve the thumbnail
            level (int): the magnification level
        Returns:
            a np.array() containing the thumbnail in rgb format
    '''
    thumbnail = slide.read_region((0, 0), level, slide.level_dimensions[level])
    return np.array(thumbnail)[:, :, :3]  # Drop the alpha channel

def apply_otsu_threshold(image):
    ''' This function applies the otsu thresholding method and returns the equivalent binary mask
        Args:
            image (np.array): rgb image on which the otsu thresholding will be applied
        Returns:
            the binary mask (np.array): a boolean mask indicating the pixels where the gray level is under the threshold
    '''
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #np.save("./graythumbnail",gray)
    thresh_val = threshold_otsu(gray)
    binary = gray < thresh_val
    return binary

def extract_patches(image, binary_mask, patch_size=(224, 224), threshold=0.9, max_patches=16):
    ''' This function extracts max_patches patches of patch_size dimensions such that 90% of 
        each patch area entails tissue based on the binary otsu mask
        
        Args:
            image (np.array) 
            binary_mask (np.array): the binary mask indicating where the tissue is present in the image 
            patch_size (tuple): the height and width of the patches
            threshold (float): the minimum percentage of tissue in each patch
            max_patches (int): the number of patches we want to get from the image
        Returns:
            a list of np.array patches
            
    '''
    patches = []
    h, w = image.shape[:2]
    ph, pw = patch_size
    step = patch_size[0] // 2  # Overlap patches by 50%
    
    for i in range(0, h - ph + 1, step):
        for j in range(0, w - pw + 1, step):
            if len(patches) >= max_patches:
                break
            patch = image[i:i+ph, j:j+pw]
            mask_patch = binary_mask[i:i+ph, j:j+pw]
            #print(mask_patch)
            
            if patch.shape[0] == ph and patch.shape[1] == pw:
                tissue_ratio = np.sum(mask_patch) / (ph * pw)
                #print(tissue_ratio)
                if tissue_ratio >= threshold:
                    #print("i,j: ",i,j)
                    patches.append((patch, mask_patch, i, j))
    return patches

def calculate_coverage(slide, patches, level, level0_binary_mask, patch_size = (224,224)):
    ''' This function calculates the total tissue coverage of the patches onto the largest magnification level
        manageable.
        
        Args:
            slide (OpenSlide object)
            patches (list of np.array)
            level (int)
            level0_binary_mask (np.array): the binary mask of the largest magnitude
            patch_size (tuple): the dimensions of the patches
    '''
    
    level_downsample = slide.level_downsamples[level] / slide.level_downsamples[1]
    
    ph, pw = patch_size
    coverage = 0

    # Get the binary mask for level 0
    #level0_thumbnail = get_thumbnail(slide, level=0)
    #level0_binary_mask = apply_otsu_threshold(level0_thumbnail)

    for _,_,i, j in patches:
        # Map the patch coordinates to level 0
        #print("i: ",i)
        
        i0 = int(i * level_downsample)
        j0 = int(j *   level_downsample)
        ph0 = int(ph * level_downsample)
        pw0 = int(pw * level_downsample)

        # Extract the corresponding area from the level 0 binary mask
        mask_patch_level0 = level0_binary_mask[i0:i0+ph0, j0:j0+pw0]
        coverage += mask_patch_level0.sum()
        #print(coverage)
    
    return coverage


def find_best_level(slide, patch_size=(224, 224), threshold=0.9, max_patches=16, level = None):
    ''' This function finds the level of magnitude onto which to apply the patches in order to maximize
        the tissue area covered by the patches (this function is meant to tackle the problem of different
        scales of tissues in the wsi)
        
        Args:
            slide (OpenSlide objet)
            patch_size (tuple)
            threshold (float): minimum percentage of tissue we want in each patch 
            max_patches (int): number of patches we want to obtain
        Returns:
            the level (int), the patches found (np.array)
        Raises:
            ValueError if we don't get enough patches at any level
    '''
    best_level = 1
    best_patches = []
    best_coverage = 0
    l0thumbnail = get_thumbnail(slide, 1)
    l0binary_mask = apply_otsu_threshold(l0thumbnail)
    
    #Iterate through every level and calculate the coverage of the patches calculated and find the best one
    for level in range(1,slide.level_count):
        thumbnail = get_thumbnail(slide, level)
        binary_mask = apply_otsu_threshold(thumbnail)
        patches = extract_patches(thumbnail, binary_mask, patch_size, threshold, max_patches)
        
        if len(patches) >= max_patches:
            coverage = calculate_coverage(slide, patches, level, l0binary_mask, patch_size)
            if coverage > best_coverage:
                best_coverage = coverage
                best_level = level
                best_patches = patches
    
    if len(best_patches) < max_patches:
        raise ValueError("Not enough patches with the required tissue content found at any level.")

    return best_level, best_patches
def get_patches_at_level(slide, patch_size=(224, 224), threshold=0.9, max_patches=16, level = 1):
    ''' This function returns "max_patches" patches of the wsi image taken in input at the fixed 'level' level of magnitude. 
        
        Args:
            slide (OpenSlide objet)
            patch_size (tuple)
            threshold (float): minimum percentage of tissue we want in each patch 
            max_patches (int): number of patches we want to obtain
            level (int): level of magnitude of the OpenSlide wsi onto which we want to retrieve the patches 
        Returns:
            the patches found (np.array)
        Raises:
            ValueError if we don't get enough patches at the provided level
    '''
    thumbnail = get_thumbnail(slide, level)
    binary_mask = apply_otsu_threshold(thumbnail)
    patches = extract_patches(thumbnail, binary_mask, patch_size, threshold, max_patches)
    if len(patches) < max_patches:
        raise ValueError("Not enough patches with the required tissue content found at any level.")
    
    return patches
    
    

def save_patches_as_numpy(patches, output_path, patient_id):
    ''' This function saves the given patches into a directory, in case the directory doesn't exist,
        it creates it automatically.
        Args:
            patches (list of np.array)
            output_path (either str or os.path)
            patient_id (str) 
    '''
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for idx, (patch, _, _, _) in enumerate(patches):
        np.save(f"{output_path}/patch_{idx}.npy", patch)