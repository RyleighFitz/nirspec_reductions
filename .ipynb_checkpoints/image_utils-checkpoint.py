################################################################################
#                      Basic Image Procesing Utilities                         #
#                                                                              #
################################################################################

from fits_dataclass import Image, ImageSet, Transform
import numpy as np
from copy import copy, deepcopy
import astroscrappy



def combine_flats(*fns):
    """ Median combine flat field images and return a master flat.
    
    INPUT:
        fns: List of flat Images
    OUTPUT:
        flat: Median combined master flat Image"""
    
    # TO DO: Update for cases with < 3 flat files (i.e. FlatCacher.cobineFlats)
        # cosmic clean flats before med combine if less than 3
    
    assert len(fns)>=1 #Make sure there are more than 1 flat image

    mflat_data = deepcopy(np.median([fn.data for fn in fns], axis=0))
    
    mflat_hdr = deepcopy(fns[0].header)
    mflat_hdr['IMAGETYP'] = 'masterflat'
    mflat_hdr['FLATFILES'] = [fn.header['FILENAME'] for fn in fns]
    
    #TO DO: Logging
    
    return Image(_data=mflat_data, header=mflat_hdr)

def header_key(key: str, value: str):
    def _pred(img: Image) -> bool:
        return img.header[key] == value
    return _pred


def cosmic_clean(img, **kwargs):
    """Implementation of astroscrappy LA Cosmics cosmic ray removal routines.
    
    INPUTS:
        data: Image : Image object to be cleanded
        **kwargs: optional kwargs for astroscrapy
        
    OUTPUT:
        cleaned: Image: Image object with CR cleaned data
    """
    
    max_iter = 3
    sig_clip = 5.0
    sig_frac = 0.3
    obj_lim = 5.0
    
    if 'readnoise' in kwargs.keys():
        readnoise=kwargs['readnoise'] 
    else:
        readnoise=10.0
        
    if 'gain' in kwargs.keys():
        gain=kwargs['gain']
    else:
        gain=2.2
        
    #TO DO: Update other kwargs if necessary (?)

    cosmicRayMask, cleanedArray = astroscrappy.detect_cosmics(img.data, pssl=0.0, gain=2.2, sigclip=sig_clip, sigfrac=sig_frac,
                                                                objlim=obj_lim, readnoise=readnoise, satlevel=np.inf, 
                                                                inmask=None, sepmed=False, cleantype='medmask', fsmode='median')
    
    
    hdr = deepcopy(img.header)
    hdr['CLEANED'] = 'yes'
    
    #TO DO: Logging
    
    return Image(_data=cleanedArray, header=hdr)


def combine_darks(*dks):
    """ Median combine dark images and return a master dark.
    
    INPUT:
        dks: List or ImageSet of dark Images
    OUTPUT:
        dark: Median combined master dark Image"""
        
    assert len(dks)>=1 #Make sure there are more than 1 flat image

    mdark_data = deepcopy(np.median([dk.data for dk in dks], axis=0))
    
    mdark_hdr = deepcopy(dks[0].header)
    mdark_hdr['IMAGETYP'] = 'masterdark'
    mdark_hdr['DARKFILES'] = [dk.header['FILENAME'] for dk in dks]
    
    #TO DO: Logging
    
    return Image(_data=mdark_data, header=mdark_hdr)


def subtract_dark(img, dark):
    """ Subtrack dark from an image.
    
    INPUT:
        img: Image: Image to be dark subtracted
        dark: dark Image
    OUTPUT:
        dimg: Image: Dark subtracted Image"""

    dimg = np.subtract(img.data, dark.data)
    
    hdr = deepcopy(img.header)
    hdr['DARKSUB'] = 'yes'
    hdr['MDARK_FILES'] = dark.header['DARKFILES']
    
    #TO DO: Logging
    
    return Image(_data=dimg, header=hdr)

def perform_darksub(*imgs):
    """ Subtrack dark from images and return a images if a dark is available. Otherwise
    return original images.
    
    INPUT:
        imgs: ImageSet: containing Images to be dark subtracted and the dark (only 1 mdark should be present!)
    OUTPUT:
        dimgs: ImageSet: Set of dark subtracted Images"""
    
    imgs = ImageSet(imgs) #TO DO: Change transform set to take an ImageSet
    dark = imgs.query(IMAGETYP = 'masterdark')
    print(len(imgs), len(dark))
    print(imgs[0].header['ITIME'])
    assert len(dark) <= 1 # Make sure no more than 1 dark is present
    if len(dark) == 0:
        return imgs
    else:
        dark = dark[0] #pull Image from ImageSet
    
        #Perform dark subtraction on object and flat images, but not on other images
        return ImageSet([subtract_dark(im, dark) if im.header['IMAGETYP']=='object' 
                     or im.header['IMAGETYP']=='masterflat' else im for im in imgs])