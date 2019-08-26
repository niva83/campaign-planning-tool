# Campaing Planning Tool (CPT): Python library for planning and designing scanning lidar measurement campaigns

## Why CPT?
Planning scanning lidar measurement campaigns is not a trivial task. There are many constraints, originating  both from the campaign site as well from the lidar technology, which a campaign planner needs to consider to derive the best possible layout of the campaign. The same can be said for configuring scanning lidars to acquire high-quality measurements.

These tasks have been typically done ad-hoc and manually, thus requiring lidar experts. However, since 2018 a work has been put to digitilize these process, making them simpler for end-users.

After almost a decade of planning and configuring scanning lidar measurement campaign, the accumulated experience and knowledge has been converted in the Campaign Planning Tool (short CPT), fascilitating the above mentioned tasks. 

**You don't need to be a scanning lidar expert anymore to design and configure scanning lidar campaigns!!!**
<br>That burden has been eliminated now, or at least that's the hope!

## What CPT is capable of doing?
The CPT provides users with a set of methods (read functions) that will allow end-users to:
* Optimize measurement positions
* Generate GIS layers which fascilitate placement of lidars 
* Optimize and synchronize trajectories for multiple lidars
* Export results in human and machine readable formats (KML, XML, YAML, etc.)
* and more ...

...and this is only the begining ! <br> <br>
For more details check out a:
* [paper that describes the CPT background](https://www.wind-energ-sci-discuss.net/wes-2019-13/)
* [presentation from the WESC conference in Cork](https://zenodo.org/record/3247797).
<br>
<br>
With every new version of the CPT library new functionalities will be aded.

## How can I get CPT?
If you don't have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your computer you should do that first.

Afterwards, copy and execute the following command in the terminal of your choice:
```
conda create -n CPT -c https://conda.windenergy.dtu.dk/channel/open -c conda-forge campaign-planning-tool gdal=2.4
```
This will create a new conda enviroment **CPT**, and download and install **campaign_planning_tool** library together with all the dependencies (see the list of libraries in 'Well deserved KUDOS goes to...'). Feel free to change the name of the enviroment to whatever name it suites  you (i.e., simply change CPT to something else).

Following the installation you need to activate newly made enviroment in the terminal:
```
source activate CPT
```
Now start the python editor of your choice, for example **jupyter**:
```
jupyter-notebook
```
Once in **jupyter** import the CPT class:
```
from campaign_planning_tool import CPT
```
and start using the CPT library.

The CPT library is fully documented so hit *help* to get a class or class method description:
```
help(CPT)
or
help(CPT.set_utm_zone)
```
## Examples 
Working with a new library is always a bit of pain, that why examples on how to use the campaign-planning-tool library are provided on a separate repo:<br>
https://github.com/niva83/campaign-planning-tool-examples
<br>
Also, the instructional videos will be available at the following YouTube channel:<br>
https://www.youtube.com/user/cadenza83/


## How to cite CPT 
If you are using CPT, you are kindly asked to cite this repository as well the paper which describes methodology which was used to develop CPT: 
```
*repository*:

*paper*:

```

## Well deserved KUDOS goes to ...
I would like to thank to awesome developers of following Python libraries that are an integrating part of the CPT:

* [whitebox](https://pypi.org/project/whitebox/)
* [srtm.py](https://github.com/tkrajina/srtm.py)
* [gdal](https://github.com/tkrajina/srtm.py)
* [geopandas](http://geopandas.org/)
* [numpy](https://www.numpy.org/)
* [pandas](https://pandas.pydata.org/)
* [pillow](https://pillow.readthedocs.io/en/stable/)
* [dicttoxml](https://pypi.org/project/dicttoxml/)
* [simplekml](https://simplekml.readthedocs.io/en/latest/)
* [pyyaml](https://pyyaml.org/)
* [matplotlib](https://matplotlib.org/)
* [jupyter](https://jupyter.org/)
* [pylint](https://www.pylint.org/)

## License
campaign-planning-tool is provided under the [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause) license.
