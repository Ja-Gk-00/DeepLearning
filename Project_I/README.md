# Project I - CNN

This is the subpage for the first project for the Deep Learning Methods classes on the WUT (Warsaw University of Technology).  
The project focused on evaluating and researching the behaviour of CNN's (Convolutional Neural Networks), for few given architectures
based on their respective articles, as well as on our own suggested architectures set up in pytorch module.  

The architectures that were comprehensly evaluated include:
- Basic CNN architecture (our own pytorch architecture)
- Optimal CNN architecture (our own pytorch architecture)
- ResNet [https://link.springer.com/chapter/10.1007/978-1-4842-6168-2_6](Reference to ResNet paper)  
- DenseNet [https://ieeexplore.ieee.org/abstract/document/8296389](Reference to DenseNet paper)  
- EfficientNet [https://link.springer.com/chapter/10.1007/978-1-4842-6168-2_10](Reference to EfficientNet paper)  
- Placeholder for possibly more architectures if there is time!  

Experiments included:
- Placeholder for experiment description 1  
- Placeholder for experiment description 2  
- Placeholder for experiment description 3  
- Placeholder for experiment description 4  

## How to set up environment and download data  
To set up work environment, user should go through the following commands:  
 1. Set up the local environment (best with python 3.10.5): `py -3.10 -m venv myevn`    
 2. Then, activate the environment: `.\myenv\Scripts\activate`  
 3. Install all requirements: `pip install -r requirements.txt`
 4. Download raw data through script from kaggle: `.\myenv\Scripts\python \data_downlader_script.py` 
 5. Run data transformation (at best keep the arguments like in the example): `.\myenv\Scripts\python .\data_transformer_script.py --format npy --output_dir Data/Data_converted`
 6. In the file `MainEvaluation.ipynb` run all the cells/modify and make custom changes to get the desired results

 ### Models saving/loading
 Because the overall training is demanding if comes to computational cost. All the trained models are saved (and should be saved) into the **Models_Pytorch_saved** in the *.pth* extention
 and into the **Models_Pickled** into the *.pkl* format, so that they can be later easily loaded for evaluation.