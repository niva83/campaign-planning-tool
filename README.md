# campaign-planning-tool: <br>Python library for planning and configuring scanning lidar measurement campaigns

## Why campaign-planning-tool?
Planning scanning lidar measurement campaigns is not a trivial task. There are many constraints, originating  both from the campaign site as well from the lidar technology, which a campaign planner needs to consider to derive the best possible layout of the campaign. The same can be said for configuring scanning lidars to acquire high-quality measurements.

These tasks have been typically done ad-hoc and manually, thus requiring lidar expertize. However, since 2018 a work has been put to digitalize these processes, making them simpler for end-users.

[After almost a decade of planning and configuring scanning lidar measurement campaigns](https://zenodo.org/record/1442592), the accumulated experience and knowledge has been converted in **campaign-planning-tool** library, fascilitating the above mentioned tasks. 

**You don't need to be a scanning lidar expert anymore to design and configure scanning lidar campaigns!!!**
<br>That burden has been eliminated now, or at least that's the hope!

## What campaign-planning-tool is capable of doing?
**campaign-planning-tool** provides users with a set of methods (read functions) that will allow them to:
* Optimize measurement positions
* Generate GIS layers which facilitate placement of lidars 
* Optimize and synchronize trajectories for multiple lidars
* Export results in human and machine readable formats (KML, XML, YAML, etc.)
* and more ...

...and this is only the beginning ! <br> <br>
For more details check out a:
* [paper that describes the ampaign-planning-tool background](https://www.wind-energ-sci-discuss.net/wes-2019-13/)
* [presentation from the WESC conference in Cork](https://zenodo.org/record/3247797).
<br>
With every new version of the **campaign-planning-tool** library new functionalities will be aded.

## How can I get campaign-planning-too?
Through DTU hosted **conda-forge**!<br>
If you don't have [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed on your computer you should install either of them first.

Afterwards, copy and execute the following command in the terminal:
```
conda create -n CPT -c https://conda.windenergy.dtu.dk/channel/open -c conda-forge campaign-planning-tool gdal=2.4
```
This will create a new conda enviroment **CPT**, and download and install **campaign-planning-tool** library together with all the dependencies. Feel free to change the name of the environment to whatever name it suites  you (i.e., simply change CPT to something else).

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
and start using the CPT library (using underscores to call library is not a mistake!).

The library is fully documented so hit *help* to get a class or class method description:
```
help(CPT)
or
help(CPT.set_utm_zone)
```
## Examples 
Working with a new library is always a bit of pain, that's why examples on how to use the campaign-planning-tool library are provided on a separate repo:<br>
https://github.com/niva83/campaign-planning-tool-examples
<br>
Also, the instructional videos will be available at the following YouTube channel:<br>
https://www.youtube.com/user/cadenza83/

## Issues, Requests, Kudos and Curses
If you have issues running campaign-planning-tool or you have requests or by any chance you want to contribute to the further development of the library please post Issues or make Pull requests on Github. 
<br>Use email communication as a last resort!!!

## How to cite campaign-planning-tool 
To continue developing and improving **campaign-planning-tool** citations are need.<br>
Therefore, if you are using **campaign-planning-tool**, you are kindly asked to cite this repository as well the paper which describes methodology which was used to develop the library (Thanks!): 
```
*repository*
Nikola Vasiljevic. (2019, July 4). campaign-planning-tool: Beta release (Version 0.1.0). 
Zenodo. http://doi.org/10.5281/zenodo.3268677

*paper*
Vasiljević, N., Vignaroli, A., Bechmann, A., and Wagner, R.: 
Digitalization of scanning lidar measurement campaign planning, 
Wind Energ. Sci. Discuss., in review, 2019. 

```

## Acknowledgement 
Well deserved kudos go to awesome developers of following Python libraries that are an integrating part of **campaign-planning-tool**:

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

as well to the members of [RECAST](http://www.recastproject.dk/project) project: [Andrea Vignaroli](https://www.dtu.dk/english/service/phonebook/person?id=94735&tab=2&qt=dtupublicationquery) (DTU), [Andreas Bechmann](https://www.dtu.dk/english/service/phonebook/person?id=20603&tab=1) (DTU), [Rozenn Wagner](https://www.dtu.dk/english/service/phonebook/person?id=38872&tab=2&qt=dtupublicationquery) (DTU) and [Morten Thøgersen](https://dk.linkedin.com/in/morten-lybech-th%C3%B8gersen-4114746) (EMD) who helped in crafting the methodology, and not to forget [Neil Davis](https://www.dtu.dk/english/service/phonebook/person?id=68826&tab=1) (DTU) who helped making the library available through **conda-forge**.

## License
campaign-planning-tool is provided under the [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause) license.


## Contact
[Nikola Vasiljević](https://www.dtu.dk/english/service/phonebook/person?id=62218&tab=2&qt=dtupublicationquery), niva@dtu.dk 
