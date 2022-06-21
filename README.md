
<div style="float: left">
    <h1> <img src="https://user-images.githubusercontent.com/47796061/97297065-6c161d80-1849-11eb-949b-390894dd41fa.png" width="100" height="100" valign="middle"> ERICA: Emulated Retinal Image CApture </h1>
</div>


## Description
ERICA is a simulation tool for generating synthetic high resolution *en face* images of the human cone mosaic based on the data capture through an adaptive optics scanning laser ophthalmoscope. Synthetic images with corresponding ground truth data are useful for testing, training and validating image processing and analysis tools, such as for cell detection, eye movement extraction and montaging. ERICA is comprised of three main modules to:

1. Generate a self-organising mosaic of cone cells
2. Simulate movements of the eye made during the image capture
3. Replicate the data capture, including the effects of diffraction, noise and residual aberrations

## Citation
If you use ERICA please cite the original paper:<br/>
Young, L.K., Smithson, H.E. Emulated retinal image capture (ERICA) to test, train and validate processing of retinal images. Sci Rep 11, 11225 (2021). https://doi.org/10.1038/s41598-021-90389-y<br/>
  
## Requirements
ERICA is written in the Python Programming language and uses the following libraries:
<br/>
numpy<br/>
scipy<br/>
yaml<br/>
numba<br/>
PIL<br/>
scikit-image<br/>
scikit-learn<br/>
astropy<br/>
<br/>
A python environment is provided (erica_environment.yml), which you can use to make sure you have all of the relevant modules and the correct versions.<br>
<br>
The specifications of your AOSLO are stored in YAML (.yml) configuration files. 
Data is stored in .fits files, which can be read in Matlab using the fitsread function. FITS allows multidimensional arrays to be stored and is also more efficient than CSV. 
Numba is used to speed up some operations.
<br/>
## How to use ERICA
ERICA comes with a series of Jupyter notebooks that go through step-by-step how images are generated. We recommend following these to understand how it works. We also include an example to generate a test dataset, which you could use as a template for your own simulations. Details about individual functions can be found in the doc strings.


<img src="https://user-images.githubusercontent.com/47796061/96901110-bdf62680-148a-11eb-98f2-ad0b82c389ac.png">

