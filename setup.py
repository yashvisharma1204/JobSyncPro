from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT = '-e .'
def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirments
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name="JobSyncPro",
    version="0.0.1",
    author="Yashvi",
    author_email="yashvi.sharma1204@gmail.com",
    description="An AI-powered resume and job description analysis tool.",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
    include_package_data=True,
    zip_safe=False,
)
