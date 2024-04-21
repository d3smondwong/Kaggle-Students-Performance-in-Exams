from setuptools import find_packages,setup
from typing import List # so you are able to return back a list under get_requirements(file_path:str)->List[str]

HYPEN_E_DOT='-e .'

# get_requirements() function will return the list of requirements in requirements.txt
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj: # file_path = 'requirements.txt'
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements] # For all line in 'requirements.txt' replace all the \n with blanks "" as .readlines() will also recognise the paragraph

        if HYPEN_E_DOT in requirements:
          requirements.remove(HYPEN_E_DOT)
    
    return requirements

#The -e option in pip install stands for "editable". 
# When you use pip install -e path/to/package, it installs the package from a local path in editable mode. 
# This means the package is installed in such a way that changes to the source code immediately affect the installed package without needing a reinstall. 
# This is useful for development on a Python package.

setup(
name='Kaggle Predict students performance in exams',
version='0.0.1',
author='Desmond Wong',
author_email='d3smondwong@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)