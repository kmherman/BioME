<img src="https://github.com/kmherman/BioME/blob/main/doc/Biomelogo.png" width="200" />__Supervised Machine Learning for Microbiome Data__
=======================================================================


BACKGROUND
----------
#####_**The problem being addressed**_
Once thought only to be pathogenic, the microorganisms living on and within an animal host (collectively known as the microbiome) are now recognized as playing critical roles in host health [[1]](#1). For example, symbiotic microorganisms found in the gastrointestinal tract, which constitute the gut microbiome, contribute to nutrient uptake [[2]](#2) and immune system maintenance [[3]](#3) that impact the host’s fitness across its lifespan. Factors that shape gut microbial communities are multifaceted and include the host’s diet [[4]](#4) and lifestage [[5]](#5). While microbial shifts have been implicated in numerous human ailments (e.g., obesity, anxiety, inflammatory bowel disease) [[6-8]](#6-8), research has thus far been limited to differentiating microbial communities between groups and less for predictive uses. Therefore, the ability for researchers and medical staff alike to be able to predict a sample's particular status based on microbial composition could further advance our understanding of host-microbiome interactions.Further, as a result of an ever expanding microbiome data availability, microbiome research lends itself to advancement with supervised machine learning [[9]](#9). With the use of supervised machine learning tools such as BioME, models trained on a dataset of samples with known labels, can be used to predict the labels of unknown samples which could be of praticular use in areas such as predicting of disease/susceptibility, sample collection site or even species from which the sample came from [[10]](#10). However, the implementation of machine learning in microbiome research might feel daunting and time consuming to those outside of the realm of data science. Further, determining which machine learning algrothisms to use can also be difficult.
To address these challenges, we developed a tool that implements and compares the performance of:
* K-Nearest Neighbor
* Neural Network
* Logistic regression
* Support Vector Machine
* Decision Tree
* Random Forest

algrothisms to perform feature selection on microbiome data and provide the most accurate model for non-data science researchers to utilize for predicting sample characteristics based on on 16S rRNA microbiome composition data. Here, we applied the tool to analyze fecal microbiome (16S rRNA gene profiles) data collected from healthy humans **(HC)** and individuals with either Crohn's disease **(CD)**, Ulcerative colitis **(UC)**, or Ileal Crohn's diease **(IC)** [[11-13]](#11-13) to display the tools capabilities.



USER PROFILES
-----
#####_**Who uses the system. What they know about the domain and computing (e.g., can browse the web, can program in Python).**_


###### Medical Clinicians
This individual can browse the web and might have no to basic Python skills, but would be uncomfortable with attempting to create their own supervised machine learning pipeline. They would be capabable of installing the tool and running BioME appropriately and to interepret the results. They aren't too sure about this and anticipate needing to phone a friend.

###### Microbiologist
This individual is very use to the data format, has intermediate skills in using Python. They have some background with bioinformatic pipelines implemented in Python such as [QIIME2](qiime2.org) [[14]](#14) and feel ready to tackle this project.

###### Ecologist
This individual has a solid background in searching the web —_but boy oh boy_— don't they love R & Rstudio and they are only begrudgingly using this tool because R has let them down— Its been too slow, and they are tired of using 10 packages to run three tests... They will give this a try, but don't think you'll be converting them to the darkside anytime soon.

*Luckily for all of the above, if they can follow installation directions and test out the system with our example, they should be able to use BioME for their microbiome classification/prediction needs!*

DATA SOURCES
---------
#####_**What data you will use and how it is structured?**_
Four open source microbiome datasets were obtained and processed in Qiita [[15]](#15). The full bioinformatic pipeline was conducted in Qiita with QIIME2 [[14]](#14) and can be  found [here](https://qiita.ucsd.edu/analysis/description/32520/).

The merged microbiome dataset consist of a count matrix in which the bacterial Operational Taxonomic Units **(OTUs)** assigned with Deblur [[16]](#16) and characterized by 16S rRNA v4 amplicon sequencing as the rows and the sample IDs from which the bacterial count information is obtained are columns. Additionally, there is metadata associated with each sample. OTUs were derived from fecal samples collected from healthy humans **(HC)** and individuals with either Crohn's disease **(CD)**, Ulcerative colitis **(UC)**, or Ileal Crohn's diease **(IC)** [[11-13]](#11-13).
Briefly, datasets were merged, rarified to 1000 sequences/sample, assigned taxonomy with a prefitted sklearn-based classifier, filtered to only include fecal samples and the assigned categories above, OTUs that were found in fewer than 10 samples were removed, and 1.) raw OTU counts and 2.) relative frequency tables in BIOM format were downloaded along with metadata for each sample. BIOM files were then converted into tsv files. Finally, a new column in the metadata was created which assigned each sample to either **HC, UC, CD, or IC**. Full collection, DNA extraction, and 16sRNA amplicon sequencing methodologies for each study can be found in the provided papers [[11-13]](#11-13).
For analysis, we used the inflammatory disease status of the individuals (**HC, CD, UC, IC**) as the response variable and count of OTUs within each sample as explanatory variables in the model building process.

_**NOTE: We started with five studies but one (ID: 1189) was removed because the samples were only from rectum, but is still shown on the Qitta analysis.**_

USE CASES
---------
#####_**Describing at least two use cases. For each, describe: (a) the objective of the user interaction; and (b) the expected interactions between the user and your system.**_

###### Medical Clinicians
The Medical Clinician has OTU count datasets from individuals with a known inflammatory bowel disease that has been linked to distruption of the fecal microbiome along with data from healthy people. They also have a growing pile of samples from individuals who they suspect have the same disease, but they haven't been diagnosed yet. If they could use BioME on their known cases, they could get results from the top model which could help them diagnose and treat paitence before their illness progresses.
They intend to use the machine to do the heavy lifting for them and use the information to move forward with validating the results, but won't go too much further with using BioME.

(**a**) The **Medical Clinician** wants to be able to determine if suspected paitents have inflammatory bowel disease so they can begin effective treatment before their health diteriates, and will use BioME to classify people.
(**b**) The **Medical Clinician** isn't worried about statistics too much, or even what the best model is, they just want to get farily accurate results quickly, so they can move forward with either further investigation into diagnosis or treatment.

###### Microbiologist
The Microbiologist is intrigued. Can BioME give them some good options for the best algrothism for their dataset? We expect our microbiologist to use BioME as a preliminary tool. Do they see something cool from the results? Is there more to look into? The microbiologist will probably take the information gleaned from BioME and dig into it further. We expect that they might even take the best suggested model and modify/customize it exactly for their data. *They aint afraid of no Python!*

(**a**) The **Microbiologist** is here for the preliminary results/confirmation. The lab intern might have mislablled some (~100) of the samples. Does say that a *Turdis turdis* sample or just a *Turtle's turds*? They don't want to have to throw out all those samples..
(**b**) The **Microbiologist** will use it to ensure the samples were labelled correctly with high confidence, and their interaction with BioME will be smooth.

###### Ecologist
The Ecologist (*after making sure there were no other options outside of using Python*) has Python set up, BioME installed, has read all of the files provided several times.

(**a**) The **Ecologist** is interested in seeing if habitat quality can be predicted based on fecal microbiome community composition in American marten (*Martes americana*). They have several hundred samples from marten living in primary undisturbed habitat and from marten from logging sites and heavy deforestation. They were lucky enough to collect even more samples from harvested marten, but don't know their habitat status. They want to determine if they can classify individuals from primary or disturbed habitat as this could be a powerful tool for conservation and management monitoring!
(**b**) The **Ecologist** will now take the data and results and proceed to go back to R to make all visualizations and any post-hoc tests.

REFERENCES
----------
<a id="1">[1]</a> **The gut microbiota — masters of host development and physiology**. Sommer, F., Bäckhed, F. . Nat Rev Microbiol 11, 227–238 (2013). https://doi.org/10.1038/nrmicro2974
<a id="2">[2]</a> LeBlanc JG, Milani C, de Giori GS, Sesma F, van Sinderen D, Ventura M. **Bacteria as vitamin suppliers to their host: a gut microbiota perspective**. Curr Opin Biotechnol. 2013 Apr;24(2):160-8. doi: 10.1016/j.copbio.2012.08.005. Epub 2012 Aug 30. PMID: 22940212.
<a id="3">[3]</a> Hooper LV, Littman DR, Macpherson AJ. **Interactions between the microbiota and the immune system**. Science. 2012;336(6086):1268-1273. doi:10.1126/science.1223490
<a id="4">[4]</a> Muegge BD, Kuczynski J, Knights D, Clemente JC, González A, Fontana L, Henrissat B, Knight R, Gordon JI. **Diet drives convergence in gut microbiome functions across mammalian phylogeny and within humans**. Science. 2011 May 20;332(6032):970-4. doi: 10.1126/science.1198719. PMID: 21596990; PMCID: PMC3303602.
<a id="5">[5]</a> McKenney EA, Rodrigo A, Yoder AD (2015). **Patterns of Gut Bacterial Colonization in Three Primate Species**. PLoS ONE 10(5): e0124618.https://doi.org/10.1371/journal.pone.0124618
<a id="6">[6]</a> Turnbaugh, P., Ley, R., Mahowald, M. et al. **An obesity-associated gut microbiome with increased capacity for energy harvest**. Nature 444, 1027–1031 (2006). https://doi.org/10.1038/nature05414
<a id="7">[7]</a> Jane A. Foster, Karen-Anne McVey Neufeld. **Gut–brain axis: how the microbiome influences anxiety and depression**.2013. Trends in Neurosciences, Volume 36, Issue 5, Pages 305-312,
ISSN 0166-2236, https://doi.org/10.1016/j.tins.2013.01.005.
<a id="8">[8]</a> Tamboli CP, Neut C, Desreumaux P, Colombel JF. **Dysbiosis in inflammatory bowel disease**. Gut. 2004;53(1):1-4. doi:10.1136/gut.53.1.1
<a id="9">[9]</a> Zhou Y-H and Gallins P. (2019). **A Review and Tutorial of Machine Learning Methods for Microbiome Host Trait Prediction**. Front. Genet. 10:579. https://doi.org/10.3389/fgene.2019.00579
<a id="10">[10]</a> Begüm D. Topçuoğlu, Nicholas A. Lesniak, Mack T. Ruffin IV, Jenna Wiens, Patrick D. Schloss. **A Framework for Effective Application of Machine Learning to Microbiome-Based Classification Problems**. mBio Jun 2020, 11 (3) e00434-20; DOI: 10.1128/mBio.00434-20
<a id="11">[11]</a> Gevers D, Kugathasan S, Denson LA, Vázquez-Baeza Y, Van Treuren W, Ren B, Schwager E, Knights D, Song SJ, Yassour M, Morgan XC, Kostic AD, Luo C, González A, McDonald D, Haberman Y, Walters T, Baker S, Rosh J, Stephens M, Heyman M, Markowitz J, Baldassano R, Griffiths A, Sylvester F, Mack D, Kim S, Crandall W, Hyams J, Huttenhower C, Knight R, Xavier RJ.**The treatment-naive microbiome in new-onset Crohn's disease. Cell Host Microbe.**. 2014 Mar 12;15(3):382-392. doi: 10.1016/j.chom.2014.02.005. PMID: 24629344; PMCID: PMC4059512. https://pubmed.ncbi.nlm.nih.gov/24629344/
<a id="12">[12]</a> Daniel McDonald, Embriette Hyde, Justine W. Debelius, James T. Morton, Antonio Gonzalez, Gail Ackermann, Alexander A. Aksenov, Bahar Behsaz, Caitriona Brennan, Yingfeng Chen, Lindsay DeRight Goldasich, Pieter C. Dorrestein, Robert R. Dunn, Ashkaan K. Fahimipour, James Gaffney, Jack A. Gilbert, Grant Gogul, Jessica L. Green, Philip Hugenholtz, Greg Humphrey, Curtis Huttenhower, Matthew A. Jackson, Stefan Janssen, Dilip V. Jeste, Lingjing Jiang, Scott T. Kelley, Dan Knights, Tomasz Kosciolek, Joshua Ladau, Jeff Leach, Clarisse Marotz, Dmitry Meleshko, Alexey V. Melnik, Jessica L. Metcalf, Hosein Mohimani, Emmanuel Montassier, Jose Navas-Molina, Tanya T. Nguyen, Shyamal Peddada, Pavel Pevzner, Katherine S. Pollard, Gholamali Rahnavard, Adam Robbins-Pianka, Naseer Sangwan, Joshua Shorenstein, Larry Smarr, Se Jin Song, Timothy Spector, Austin D. Swafford, Varykina G. Thackray, Luke R. Thompson, Anupriya Tripathi, Yoshiki Vázquez-Baeza, Alison Vrbanac, Paul Wischmeyer, Elaine Wolfe, Qiyun Zhu, The American Gut Consortium, Rob Knight. **American Gut: an Open Platform for Citizen Science Microbiome Research**. mSystems May 2018, 3 (3) e00031-18; DOI: 10.1128/mSystems.00031-18 https://msystems.asm.org/content/3/3/e00031-18
<a id="13">[13]</a> Halfvarson, J., Brislawn, C., Lamendella, R. et al. **Dynamics of the human gut microbiome in inflammatory bowel disease**.  Nat Microbiol 2, 17004 (2017). https://doi.org/10.1038/nmicrobiol.2017.4
<a id="14">[14]</a>  Bolyen E, Rideout JR, Dillon MR, Bokulich NA, Abnet CC, Al-Ghalith GA, Alexander H, Alm EJ, Arumugam M, Asnicar F, Bai Y, Bisanz JE, Bittinger K, Brejnrod A, Brislawn CJ, Brown CT, Callahan BJ, Caraballo-Rodríguez AM, Chase J, Cope EK, Da Silva R, Diener C, Dorrestein PC, Douglas GM, Durall DM, Duvallet C, Edwardson CF, Ernst M, Estaki M, Fouquier J, Gauglitz JM, Gibbons SM, Gibson DL, Gonzalez A, Gorlick K, Guo J, Hillmann B, Holmes S, Holste H, Huttenhower C, Huttley GA, Janssen S, Jarmusch AK, Jiang L, Kaehler BD, Kang KB, Keefe CR, Keim P, Kelley ST, Knights D, Koester I, Kosciolek T, Kreps J, Langille MGI, Lee J, Ley R, Liu YX, Loftfield E, Lozupone C, Maher M, Marotz C, Martin BD, McDonald D, McIver LJ, Melnik AV, Metcalf JL, Morgan SC, Morton JT, Naimey AT, Navas-Molina JA, Nothias LF, Orchanian SB, Pearson T, Peoples SL, Petras D, Preuss ML, Pruesse E, Rasmussen LB, Rivers A, Robeson MS, Rosenthal P, Segata N, Shaffer M, Shiffer A, Sinha R, Song SJ, Spear JR, Swafford AD, Thompson LR, Torres PJ, Trinh P, Tripathi A, Turnbaugh PJ, Ul-Hasan S, van der Hooft JJJ, Vargas F, Vázquez-Baeza Y, Vogtmann E, von Hippel M, Walters W, Wan Y, Wang M, Warren J, Weber KC, Williamson CHD, Willis AD, Xu ZZ, Zaneveld JR, Zhang Y, Zhu Q, Knight R, and Caporaso JG. **Reproducible, interactive, scalable and extensible microbiome data science using QIIME 2**. 2019. Nature Biotechnology 37: 852–857. https://doi.org/10.1038/s41587-019-0209-9
[QIIME2 website](qiime2.org)
 <a id="15">[15]</a> Antonio Gonzalez, Jose A. Navas-Molina, Tomasz Kosciolek, Daniel McDonald, Yoshiki Vázquez-Baeza, Gail Ackermann, Jeff DeReus, Stefan Janssen, Austin D. Swafford, Stephanie B. Orchanian, Jon G. Sanders, Joshua Shorenstein, Hannes Holste, Semar Petrus, Adam Robbins-Pianka, Colin J. Brislawn, Mingxun Wang, Jai Ram Rideout, Evan Bolyen, Matthew Dillon, J. Gregory Caporaso, Pieter C. Dorrestein & Rob Knight. **Qiita: rapid, web-enabled microbiome meta-analysis**. Nature Methods, volume 15, pages 796–798 (2018); https://doi.org/10.1038/s41592-018-0141-9
[Qiita website](https://qiita.ucsd.edu/)
<a id="16">[16]</a>  Amnon Amir, Daniel McDonald, Jose A. Navas-Molina, Evguenia Kopylova, James T. Morton, Zhenjiang Zech Xu, Eric P. Kightley, Luke R. Thompson, Embriette R. Hyde, Antonio Gonzalez, Rob Knight. **Deblur Rapidly Resolves Single-Nucleotide Community Sequence Patterns**. mSystems Mar 2017, 2 (2) e00191-16; DOI: 10.1128/mSystems.00191-16 https://msystems.asm.org/content/2/2/e00191-16
