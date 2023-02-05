# image-super-resolution



**MODULES**

`test.py` Module for generating HQ test images from the SRResNet model.
`annotation.py` Module for generating the training annotation file.
`plot.py` Module for reading json logs files and plotting the training curves in the report.
`srresnet_project.py` Modifed configuration file for SRResNet.
`esrgan_project_psnr.py` Modified configuration file for RRDBNet.
`esrgan_project.py` Modified configuration file for ESRGAN module (for future work).


`/project/best_model_HQimages/iter_285.pth`	Checkpoint for the best SRResNet model.
`/project/best_model_HQimages/srresnet_285k.zip`	Zipped file containing 400 HQ images from the best model.

`/project/data/train_ann.txt`			Annotation file.

`/project/srresnet_logs/*`			Training logs (txt and json) for SRResNet.
`/project/srresnet_logs/srresnet.json`		Complete training logs used to plot the training curves for SRResNet.

`/project/rrdbnet_logs/*`			Training logs (txt and json) for RRDBNet.
`/project/rrdbnet_logs/rrdbnet.json`		Complete training logs used to plot the training curves for RRDBNet.
 

Links to Checkpoint File and HQ images:  

https://drive.google.com/file/d/1Ffcq7V21e2C4iK5Kb1q1Qi_TSaHVTiTV/view?usp=sharing  

https://drive.google.com/file/d/1cUQm1J2kTorAcsqoXzRvMB75HSZrKVrn/view?usp=sharing


**SETUP**

`cd image_super_resolution/project`

Create and activate a conda virtual environment.  
`conda create --name cv python=3.8 -y`  
`conda activate cv`

Install the GPU version of PyTorch and Torchvision.  
`conda install pytorch=1.11.0 torchvision cudatoolkit=10.2 -c pytorch`

Install MMCV for GPU.
`pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html`  

Install MMEditing.  
`git clone https://github.com/open-mmlab/mmediting.git`
`cd mmediting`
`pip install -v -e`

Go back to the current directory.  
`cd ..`

Move all directories and files in this package to the current directory.
The directory structure will be:

```
./best_model_HQimages/
	iter_285.pth
	srresnet_285k.zip
./data/
	test/
		HQ/
		LQ/
	train/
		GT/
		LQ/
	val/
		GT/
		LQ/
	train_ann.txt
./rrdbnet_logs/
	rrdbnet.json
	*
./srresnet_logs/
	srresnet.json
	*
./mmediting/
	*
esrgan_project.py
esrgan_project_psnr.py
srresnet_project.py
annotation.py
plot.py
test.py
```