# Databricks notebook source
# MAGIC %md # Processing genomic data in Spark
# MAGIC 
# MAGIC *You can import this notebook from [here](https://github.com/evodify/genomic-analyses_in_apache-spark/tree/master/vcf_filtering_tutorial).*
# MAGIC 
# MAGIC ## [Big Data: Astronomical or Genomical?](http://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002195)
# MAGIC 
# MAGIC ### Abstract
# MAGIC 
# MAGIC Genomics is a Big Data science and is going to get much bigger, very soon, but it is not known whether the needs of genomics will exceed other Big Data domains. Projecting to the year 2025, we compared genomics with three other major generators of Big Data: astronomy, YouTube, and Twitter. Our estimates show that genomics is a “four-headed beast”—it is either on par with or the most demanding of the domains analyzed here in terms of data acquisition, storage, distribution, and analysis. We discuss aspects of new technologies that will need to be developed to rise up and meet the computational challenges that genomics poses for the near future. Now is the time for concerted, community-wide planning for the “genomical” challenges of the next decade.
# MAGIC 
# MAGIC <img src="http://journals.plos.org/plosbiology/article/figure/image?size=large&id=10.1371/journal.pbio.1002195.g001" width="800">
# MAGIC 
# MAGIC The plot shows the growth of DNA sequencing both in the total number of human genomes sequenced (left axis) as well as the worldwide annual sequencing capacity (right axis: Tera-basepairs (Tbp), Peta-basepairs (Pbp), Exa-basepairs (Ebp), Zetta-basepairs (Zbps)).
# MAGIC 
# MAGIC 
# MAGIC <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Historic_cost_of_sequencing_a_human_genome.svg/800px-Historic_cost_of_sequencing_a_human_genome.svg.png" width="800">

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## DNA sequencing
# MAGIC 
# MAGIC <img src="https://www.yourgenome.org/sites/default/files/illustrations/process/physical_mapping_STS_yourgenome.png" width="800">

# COMMAND ----------

# MAGIC %md
# MAGIC ## Alligning millions of small DNA sequences (reads) to a reference genome 
# MAGIC 
# MAGIC <img src="https://github.com/evodify/genomic-analyses_in_apache-spark/raw/master/vcf_filtering_tutorial/genome_read_mapping.png" width="1000">

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # VCF
# MAGIC 
# MAGIC <img src="https://hail.is/docs/stable/_images/hail-vds-rep.png" width="600">

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC // This allows easy embedding of publicly available information into any other notebook
# MAGIC // when viewing in git-book just ignore this block - you may have to manually chase the URL in frameIt("URL").
# MAGIC // Example usage:
# MAGIC // displayHTML(frameIt("https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation#Topics_in_LDA",250))
# MAGIC def frameIt( u:String, h:Int ) : String = {
# MAGIC       """<iframe 
# MAGIC  src=""""+ u+""""
# MAGIC  width="95%" height="""" + h + """"
# MAGIC  sandbox>
# MAGIC   <p>
# MAGIC     <a href="http://spark.apache.org/docs/latest/index.html">
# MAGIC       Fallback link for browsers that, unlikely, don't support frames
# MAGIC     </a>
# MAGIC   </p>
# MAGIC </iframe>"""
# MAGIC    }
# MAGIC displayHTML(frameIt("https://en.wikipedia.org/wiki/Variant_Call_Format",500))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Hail
# MAGIC 
# MAGIC [Hail](https://hail.is) is an open-source, scalable framework for exploring and analyzing genomic data. Its functionality is exposed through **Python** and backed by distributed algorithms built on top of **Apache Spark** to efficiently analyze gigabyte-scale data on a laptop or terabyte-scale data on a cluster, without the need to manually chop up data or manage job failures. Users can script pipelines or explore data interactively through **Jupyter notebooks** that flow between Hail with methods for genomics, *PySpark* with scalable *SQL* and *machine learning algorithms*, and *pandas* with *scikit-learn* and *Matplotlib* for results that fit on one machine. Hail also provides a flexible domain language to express complex quality control and analysis pipelines with concise, readable code.
# MAGIC 
# MAGIC #### Scaling Genetic Data Analysis with Apache Spark
# MAGIC [![Scaling Genetic Data Analysis with Apache Spark](http://img.youtube.com/vi/pyeQusIN5Ao/0.jpg)](https://www.youtube.com/embed/pyeQusIN5Ao)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## This Notebook is based on the tutorial [Analyzing 1000 Genomes with Spark and Hail](https://docs.databricks.com/spark/latest/training/1000-genomes.html)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Cluster setup
# MAGIC 
# MAGIC First download Hail's Python and Java libraries to your computer:
# MAGIC 
# MAGIC https://storage.googleapis.com/hail-common/hail-tutorial-databricks.jar
# MAGIC 
# MAGIC https://storage.googleapis.com/hail-common/hail-devel-py2.7-databricks.egg
# MAGIC 
# MAGIC Then on the Databricks interface, navigate to `Workspace > Users > Username` and select `Import` from the Username drop-down menu. At the bottom of `Import Notebooks` window, click the link in `(To import a library, such as a jar or egg,`_`click here`_`)`.  Upload both the .jar and .egg files using this interface, using any names you like. Make sure that the option `Attach automatically to all clusters` is checked in the success dialog.
# MAGIC 
# MAGIC Next click the `Clusters` icon on the left sidebar and then `+Create Cluster`. For `Apache Spark Version`, select `Spark 2.0 (Auto-updating, Scala 2.11)`. Note that Hail won't work with Scala 2.10! In the Databricks cluster creation dialog, click `Show advanced settings` at bottom and then on the `Spark` tab, and paste the text below into the `Spark config` box.
# MAGIC 
# MAGIC ```
# MAGIC spark.hadoop.io.compression.codecs org.apache.hadoop.io.compress.DefaultCodec,is.hail.io.compress.BGzipCodec,org.apache.hadoop.io.compress.GzipCodec
# MAGIC spark.sql.files.openCostInBytes 1099511627776
# MAGIC spark.sql.files.maxPartitionBytes 1099511627776
# MAGIC spark.hadoop.mapreduce.input.fileinputformat.split.minsize 1099511627776
# MAGIC spark.hadoop.parquet.block.size 1099511627776```
# MAGIC Start the cluster and attach this notebook to it by clicking on your cluster name in menu `Detached` at the top left of this workbook. Now you're ready to Hail!

# COMMAND ----------

from hail import *
hc = HailContext(sc)

# COMMAND ----------

# MAGIC %md Let's import some Python libraries for use throughout the tutorial.

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from math import log, isnan
import seaborn

# COMMAND ----------

# MAGIC %md ## Import data
# MAGIC 
# MAGIC We must first import variant data into Hail's internal format of Variant Dataset (VDS). We use the [import_vcf](https://hail.is/hail/hail.HailContext.html#hail.HailContext.import_vcf) method on [HailContext](https://hail.is/hail/hail.HailContext.html) to load a VCF file into Hail.
# MAGIC 
# MAGIC It is recommended to load a block-compressed VCF (`.vcf.bgz`) which enables Hail to read the file in parallel. Reading files that have not been block-compressed (`.vcf`, `.vcf.gz`) is _significantly_ slower and should be avoided (though often `.vcf.gz` files are in fact block-compressed, in which case renaming to `.vcf.bgz` solves the problem).
# MAGIC 
# MAGIC Unfortunately, I was not able to load `.bgz` compressed file. It worked for the existing human dataset, which is bgz compressed, but not for my data (See below).
# MAGIC 
# MAGIC ### Cbp data
# MAGIC As a training data set, I use a subset of 1% from my unpublished genomic data on [*Capsella bursa-pastoris*](https://en.wikipedia.org/wiki/Capsella_bursa-pastoris) (hereafter Cbp).
# MAGIC 
# MAGIC ### Human data
# MAGIC 
# MAGIC You can use publicly avaliable human data by uncommenting the three line below. Jump to Cmd 19.

# COMMAND ----------

# MAGIC %md ## Download the data

# COMMAND ----------

# MAGIC %sh
# MAGIC wget https://github.com/evodify/genomic-analyses_in_apache-spark/raw/master/vcf_filtering_tutorial/Cbp31_SNPs_test0.01.vcf.gz # download the test VCF file
# MAGIC wget https://github.com/evodify/genomic-analyses_in_apache-spark/raw/master/vcf_filtering_tutorial/Cbp31_annot.csv # download the test annotation file
# MAGIC gunzip Cbp31_SNPs_test0.01.vcf.gz # uncomress the gzip file.

# COMMAND ----------

# MAGIC %sh pwd && ls -l

# COMMAND ----------

# MAGIC %md ## Move the downloaded data to DBFS

# COMMAND ----------

dbutils.fs.mkdirs("dbfs:/FileStore/tables/Cbp") # create a new directory for our data files

# COMMAND ----------

dbutils.fs.mv("file:/databricks/driver/Cbp31_SNPs_test0.01.vcf", "dbfs:/FileStore/tables/Cbp") # move
dbutils.fs.cp("file:/databricks/driver/Cbp31_annot.csv", "dbfs:/FileStore/tables/Cbp") # copy because we will need non-dbfs file for R later

# COMMAND ----------

dbutils.fs.ls("dbfs:/FileStore/tables/Cbp/")

# COMMAND ----------

# MAGIC %md ## Start processing

# COMMAND ----------

vcf_path = '/FileStore/tables/Cbp/Cbp31_SNPs_test0.01.vcf'
annotation_path = '/FileStore/tables/Cbp/Cbp31_annot.csv'

# comment out the line above and uncommend the lines below for human data

# vcf_path = '/databricks-datasets/hail/data-001/1kg_sample.vcf.bgz'
# annotation_path = '/databricks-datasets/hail/data-001/1kg_annotations.txt'
# purcell_5k_path = '/databricks-datasets/hail/data-001/purcell5k.interval_list'

# COMMAND ----------

vds = hc.import_vcf(vcf_path) # bgz import fails even with force_bgz=True. I compressed my files with bgzip from https://github.com/samtools/tabix .

# COMMAND ----------

# MAGIC %md This method produced a [VariantDataset](https://hail.is/hail/hail.VariantDataset.html), Hail's primary representation of genomic data. Following that link to Hail's python API documentation will let you see the myriad methods it offers.  We will use but a few of them in this tutorial. We next use its [split_multi](https://hail.is/hail/hail.VariantDataset.html#hail.VariantDataset.split_multi) to split multi-allelic variants into biallelic variants. For example, the variant `1:1000:A:T,C` would become two variants: `1:1000:A:T` and `1:1000:A:C`.

# COMMAND ----------

vds = vds.split_multi()

# COMMAND ----------

# MAGIC %md We next use the [annotate_samples_table](https://hail.is/hail/hail.VariantDataset.html#hail.VariantDataset.annotate_samples_table) method to load phenotypic information on each sample from the sample annotations file.
# MAGIC 
# MAGIC Here `annotation_path` refers to the sample annotation data file, whose first few lines are:
# MAGIC ```
# MAGIC Sample Population
# MAGIC DL174     ASI
# MAGIC GY37      ASI
# MAGIC HJC419    ASI
# MAGIC 12.4      EUR
# MAGIC 13.16     EUR
# MAGIC 16.9      EUR
# MAGIC BEL5      EUR
# MAGIC ```
# MAGIC 
# MAGIC The `root` argument says where to put this data. For sample annotations, the root must start with `sa` followed by a `.` and the rest is up to you, so let's use `sa.myAnnot`.
# MAGIC 
# MAGIC The `sample_expr` argument indicates that the sample ID is in column `Sample`.
# MAGIC 
# MAGIC The object `TextTableConfig` allows users to provide information about column data types, header existence, comment characters, and field delimiters. For example,  'Population: Boolean'.  `impute=True` will infer column types automatically.

# COMMAND ----------

dbutils.fs.head(annotation_path)

# COMMAND ----------

vds = vds.annotate_samples_table(annotation_path,
                                 root='sa.myAnnot',
                                 sample_expr='Sample',
                                 config=TextTableConfig(impute=True))

# COMMAND ----------

# MAGIC %md Lastly, we'll [write](https://hail.is/hail/hail.VariantDataset.html#hail.VariantDataset.write) the dataset to disk so that all future computations begin by reading in the fast VDS rather than the slow VCF.

# COMMAND ----------

out_path = '/cbp.vds'
vds.write(out_path, overwrite=True)

# COMMAND ----------

# MAGIC %md ## Start exploring
# MAGIC 
# MAGIC Now we're ready to start exploring! We will read back in the VDS we wrote to disk:

# COMMAND ----------

vds = hc.read(out_path)

# COMMAND ----------

# MAGIC %md First, we'll print some statistics about the size of the dataset using [count](https://hail.is/hail/hail.VariantDataset.html#hail.VariantDataset.count):

# COMMAND ----------

print(vds.count())

# COMMAND ----------

# MAGIC %md If the Boolean parameter `genotypes` is set to `True`, the overall call rate across all genotypes is computed as well:

# COMMAND ----------

vds.count(genotypes=True)

# COMMAND ----------

# MAGIC %md So the call rate before any QC filtering is about 84.67%.
# MAGIC 
# MAGIC Let's print the types of all annotations.
# MAGIC 
# MAGIC Variant annotations:

# COMMAND ----------

print(vds.variant_schema)

# COMMAND ----------

# MAGIC %md Sample annotations:

# COMMAND ----------

print(vds.sample_schema)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC  We have just printed the sample and variant structure above. Global structure is empty.
# MAGIC  
# MAGIC  To recall the structure:
# MAGIC 
# MAGIC <img src="https://hail.is/docs/stable/_images/hail-vds-rep.png" width="600">
# MAGIC 
# MAGIC Also, note the annotations imported from the original VCF, as well as the sample annotations added above. Notice how those six sample annotations loaded above are nested inside `sa.structure` as defined by the `root` option in [annotate_samples_table](https://hail.is/hail/hail.VariantDataset.html#hail.VariantDataset.annotate_samples_table).

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Next we'll add some global annotations including the list of populations that are present in our dataset and the number of samples in each population, using the Hail expression language and the [query_samples](https://hail.is/hail/hail.VariantDataset.html#hail.VariantDataset.query_samples) method. The codings are:
# MAGIC 
# MAGIC   - ASI = Asian
# MAGIC   - EUR = European
# MAGIC   - ME = Middle Eastern
# MAGIC 
# MAGIC We'll first build up a list of query expressions, then evaluate them all at once to save time.

# COMMAND ----------

expressions = ['samples.map(s => sa.myAnnot.Population).collect().toSet']
queries = vds.query_samples(expressions)

print('populations = %s' % queries[0])
print('total samples = %s' % vds.num_samples)

# COMMAND ----------

# MAGIC %md Now it's easy to count samples by population using the [counter](https://hail.is/expr_lang.html#counter) aggregator:

# COMMAND ----------

counter = vds.query_samples('samples.map(s => sa.myAnnot.Population).counter()')[0]
for x in counter:
    print('population %s found %s times' % (x.key, x.count))

# COMMAND ----------

# MAGIC %md ## Quality control (QC)
# MAGIC 
# MAGIC VCF file contains many annotations scores that define the quality of genotypes as well as quality of a variant.
# MAGIC 
# MAGIC ### Filter genotypes
# MAGIC 
# MAGIC Let's filter genotypes based on genotype quality (GQ) and read coverage (DP).
# MAGIC 
# MAGIC Here `g` is genotype, `v` is variant, `s` is sample, and annotations are accessible via `va`, `sa`, and `global`. 

# COMMAND ----------

# MAGIC %scala
# MAGIC 
# MAGIC // This allows easy embedding of publicly available information into any other notebook
# MAGIC // when viewing in git-book just ignore this block - you may have to manually chase the URL in frameIt("URL").
# MAGIC // Example usage:
# MAGIC // displayHTML(frameIt("https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation#Topics_in_LDA",250))
# MAGIC def frameIt( u:String, h:Int ) : String = {
# MAGIC       """<iframe 
# MAGIC  src=""""+ u+""""
# MAGIC  width="95%" height="""" + h + """"
# MAGIC  sandbox>
# MAGIC   <p>
# MAGIC     <a href="http://spark.apache.org/docs/latest/index.html">
# MAGIC       Fallback link for browsers that, unlikely, don't support frames
# MAGIC     </a>
# MAGIC   </p>
# MAGIC </iframe>"""
# MAGIC    }
# MAGIC displayHTML(frameIt("https://en.wikipedia.org/wiki/Phred_quality_score",500))

# COMMAND ----------

filter_condition_gDP_gGQ = 'g.dp >= 10 && g.gq >= 20'
vds_gDP_gGQ = vds.filter_genotypes(filter_condition_gDP_gGQ)

# COMMAND ----------

vds_gDP_gGQ.count(genotypes=True)

# COMMAND ----------

# MAGIC %md Now the call rate is about 50%, so nearly 35% of genotypes failed the filter. Filtering out a genotype is equivalent to setting the genotype call to missing.
# MAGIC 
# MAGIC Having removed suspect genotypes, let's next remove variants with low call rate and then calculate summary statistics per sample with the [sample_qc](https://hail.is/hail/hail.VariantDataset.html#hail.VariantDataset.sample_qc) method.

# COMMAND ----------

vds_gDP_gGQ_vCR = (vds_gDP_gGQ
    .filter_variants_expr('gs.fraction(g => g.isCalled) >= 0.50')
    .sample_qc())

# COMMAND ----------

# MAGIC %md
# MAGIC Check how many variants retained after filtering.

# COMMAND ----------

vds_gDP_gGQ_vCR.count(genotypes=True)

# COMMAND ----------

# MAGIC %md ### Filter samples

# COMMAND ----------

# MAGIC %md The call rate for each variant is calculated using the `fraction` [aggregable](https://hail.is/expr_lang.html#aggregables) on the genotypes `gs`. [sample_qc](https://hail.is/hail/hail.VariantDataset.html#hail.VariantDataset.sample_qc) adds a number of statistics to sample annotations:

# COMMAND ----------

print(vds_gDP_gGQ_vCR.sample_schema)

# COMMAND ----------

# MAGIC %md Let's export these sample annotations to a text file and take a look at them:

# COMMAND ----------

vds_gDP_gGQ_vCR.export_samples('file:///sampleqc.txt', 'Sample = s.id, sa.qc.*')

# COMMAND ----------

# MAGIC %sh
# MAGIC head /sampleqc.txt | cut -f 1-8 | column -t

# COMMAND ----------

# MAGIC %md We can further analyze these results locally using Python's [matplotlib](http://matplotlib.org/) library. Below is an example plot of three variables (call rate, mean depth and mean GQ), along with the code that generate the plot.

# COMMAND ----------

sampleqc_table = vds_gDP_gGQ_vCR.samples_keytable().to_pandas()

plt.clf() # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
plt.figure(figsize=(7,4)) # figure size in inches; change according to your screen size and resolution

plt.subplot(1, 3, 1)
plt.hist(sampleqc_table["sa.qc.callRate"], bins=np.arange(0.4, 1.01, .05))
plt.xlabel("Call Rate")
plt.ylabel("Frequency")
plt.xlim(0.4, 1)
plt.axvline(.50, color='r')

plt.subplot(1, 3, 2)
plt.hist(sampleqc_table["sa.qc.dpMean"], bins=np.arange(20, 80, 10))
plt.xlabel("Mean depth")
plt.ylabel("Frequency")
plt.xlim(10, 80)
plt.axvline(10, color='r')

plt.subplot(1, 3, 3)
plt.hist(sampleqc_table["sa.qc.gqMean"], bins=np.arange(20, 100, 10))
plt.xlabel("Mean Sample GQ")
plt.ylabel("Frequency")
plt.axvline(30, color = 'r')
plt.xlim(0, 100)

plt.tight_layout()
plt.show()
display()

# COMMAND ----------

# MAGIC %md
# MAGIC You can remove samples that are outliers in the plots above, where cutoffs are given by the red lines. But there are no outliers here. If we had to filter, we could do this step:

# COMMAND ----------

vds_gDP_gGQ_vCR_sDP_sGT = (vds_gDP_gGQ_vCR
    .annotate_samples_vds(vds_gDP_gGQ_vCR, code = 'sa.qc = vds.qc' )
    .filter_samples_expr('sa.qc.dpMean > 0.50 && sa.qc.dpMean >=10 && sa.qc.gqMean >= 20 '))

# COMMAND ----------

# MAGIC %md As before, we can count the number of samples that remain in the dataset after filtering. (But nothing has been filtered out here)

# COMMAND ----------

vds_gDP_gGQ_vCR_sDP_sGT.count(genotypes=True)

# COMMAND ----------

# MAGIC %md ### Filter variants
# MAGIC 
# MAGIC We now have `vds_gDP_gGQ_vCR_sDP_sGT`, a VDS where low-quality genotypes and samples have been removed.
# MAGIC 
# MAGIC Let's use the [variant_qc](https://hail.is/hail/hail.VariantDataset.html#hail.VariantDataset.variant_qc) method to start exploring variant metrics:

# COMMAND ----------

vds_gDP_gGQ_vCR_sDP_sGT = vds_gDP_gGQ_vCR_sDP_sGT.variant_qc()
print(vds_gDP_gGQ_vCR_sDP_sGT.variant_schema)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC Next, we will filter variants following the [Best Practices GATK recommendations](https://gatkforums.broadinstitute.org/gatk/discussion/2806/howto-apply-hard-filters-to-a-call-set).
# MAGIC 
# MAGIC These recommendations are for human data, but our data is not human and the distribution of quality statistics will differ from expected for human. (Explain why it is so is beyond the scope of this tutorial.)
# MAGIC 
# MAGIC Let's have a look at the distribution of different variant quality statistics:
# MAGIC 
# MAGIC - QD - variant confidence standardized by depth.
# MAGIC 
# MAGIC   This annotation puts the variant confidence QUAL score into perspective by normalizing for the amount of coverage available. Because each read contributes a little to the QUAL score, variants in region with deep coverage can have artificially inflated QUAL scores, giving the impression that the call is supported by more evidence than it really is. To compensate for this, we normalize the variant confidence by depth, which gives us a more objective picture of how well supported the call is.
# MAGIC    
# MAGIC - MQ - Mapping quality of a SNP.
# MAGIC 
# MAGIC - FS - strand bias in support for REF vs ALT allele calls.
# MAGIC 
# MAGIC   Strand bias is a type of sequencing bias in which one DNA strand is favored over the other, which can result in incorrect evaluation of the amount of evidence observed for one allele vs. the other. The FisherStrand annotation is one of several methods that aims to evaluate whether there is strand bias in the data. It uses Fisher's Exact Test to determine if there is strand bias between forward and reverse strands for the reference or alternate allele. The output is a Phred-scaled p-value. The higher the output value, the more likely there is to be bias. More bias is indicative of false positive calls.
# MAGIC   
# MAGIC - SOR - sequencing bias in which one DNA strand is favored over the other
# MAGIC    
# MAGIC    Strand bias is a type of sequencing bias in which one DNA strand is favored over the other, which can result in incorrect evaluation of the amount of evidence observed for one allele vs. the other. It is used to determine if there is strand bias between forward and reverse strands for the reference or alternate allele. The reported value is ln-scaled.
# MAGIC    
# MAGIC - MQRankSum - Rank sum test for mapping qualities of REF vs. ALT reads.
# MAGIC    
# MAGIC    This variant-level annotation compares the mapping qualities of the reads supporting the reference allele with those supporting the alternate allele. The ideal result is a value close to zero, which indicates there is little to no difference. A negative value indicates that the reads supporting the alternate allele have lower mapping quality scores than those supporting the reference allele. Conversely, a positive value indicates that the reads supporting the alternate allele have higher mapping quality scores than those supporting the reference allele.
# MAGIC 
# MAGIC - ReadPosRankSum - do all the reads support a SNP call tend to be near the end of a read.
# MAGIC    
# MAGIC    The ideal result is a value close to zero, which indicates there is little to no difference in where the alleles are found relative to the ends of reads. A negative value indicates that the alternate allele is found at the ends of reads more often than the reference allele. Conversely, a positive value indicates that the reference allele is found at the ends of reads more often than the alternate allele. 

# COMMAND ----------

# MAGIC %md
# MAGIC We've once again used matplotlib to make histograms of these siz summary statistics.

# COMMAND ----------

variantqc_table = vds_gDP_gGQ_vCR_sDP_sGT.variants_keytable().to_pandas()

plt.clf() # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
plt.figure(figsize=(7,7)) # figure size in inches; change according to your screen size and resolution

plt.subplot(3, 2, 1)
variantgq_MQ = variantqc_table["va.info.MQ"]
plt.hist(variantgq_MQ.dropna(), bins = np.arange(0, 101, 2)) # It is important to add dropna() to skip NA values. Otherwise, the script won't work.
plt.xlabel("MQ")
plt.ylabel("Frequency")
plt.xlim(0, 100)
plt.axvline(30, color = 'r')

plt.subplot(3, 2, 2)
variantgq_SOR = variantqc_table["va.info.SOR"]
plt.hist(variantgq_SOR.dropna(), bins = np.arange(0, 8, 0.2))
plt.xlabel("SOR")
plt.ylabel("Frequency")
plt.xlim(0, 8)
plt.axvline(4, color = 'r')

plt.subplot(3, 2, 3)
variantgq_QD = variantqc_table["va.info.QD"]
plt.hist(variantgq_QD.dropna(), bins = np.arange(0, 40, 1))
plt.xlabel("QD")
plt.ylabel("Frequency")
plt.xlim(0, 40)
plt.axvline(2, color = 'r')

plt.subplot(3, 2, 4)
variantgq_FS = variantqc_table["va.info.FS"]
plt.hist(variantgq_FS.dropna(), bins = np.arange(0, 100, 2))
plt.xlabel("FS")
plt.ylabel("Frequency")
plt.xlim(0, 100)
plt.axvline(60, color = 'r')

plt.subplot(3, 2, 5)
variantgq_MQRankSum = variantqc_table["va.info.MQRankSum"]
plt.hist(variantgq_MQRankSum.dropna(), bins = np.arange(-20, 20, 1))
plt.xlabel("MQRankSum")
plt.ylabel("Frequency")
plt.xlim(-25, 25)
plt.axvline(-20, color = 'r')

plt.subplot(3, 2, 6)
variantgq_ReadPosRankSum = variantqc_table["va.info.ReadPosRankSum"]
plt.hist(variantgq_ReadPosRankSum.dropna(), bins = np.arange(-20, 20, 0.5))
plt.xlabel("ReadPosRankSum")
plt.ylabel("Frequency")
plt.xlim(-12, 12)
plt.axvline(-10, color = 'r')
plt.axvline(10, color = 'r')

plt.tight_layout()
plt.show()
display()

# COMMAND ----------

# MAGIC %md 
# MAGIC Lastly we use the [filter_variants_expr](https://hail.is/hail/hail.VariantDataset.html#hail.VariantDataset.filter_variants_expr) method to keep only those variants that meet the cut-off requirements (red lines in the plots above).

# COMMAND ----------

vds_gDP_gGQ_vCR_sDP_sGT_vFilter = vds_gDP_gGQ_vCR_sDP_sGT.filter_variants_expr('va.info.MQ >= 30.00 && va.info.SOR <= 4.000 && va.info.QD >= 2.00 && va.info.FS <= 60.000 && va.info.MQRankSum >= -20.000 && va.info.ReadPosRankSum >= -10.000 && va.info.ReadPosRankSum <= 10.000')
print('variants before filtering: %d' % vds_gDP_gGQ_vCR_sDP_sGT.count_variants())
print('variants after filtering: %d' % vds_gDP_gGQ_vCR_sDP_sGT_vFilter.count_variants())

# COMMAND ----------

# MAGIC %md Verify the filtering results with plots:

# COMMAND ----------

variantqc_table = vds_gDP_gGQ_vCR_sDP_sGT_vFilter.variants_keytable().to_pandas()

plt.clf() # clears the entire current figure with all its axes, but leaves the window opened, such that it may be reused for other plots.
plt.figure(figsize=(7,7)) # figure size in inches; change according to your screen size and resolution

plt.subplot(3, 2, 1)
variantgq_MQ = variantqc_table["va.info.MQ"]
plt.hist(variantgq_MQ.dropna(), bins = np.arange(0, 101, 2))
plt.xlabel("MQ")
plt.ylabel("Frequency")
plt.xlim(0, 100)
plt.axvline(30, color = 'r')

plt.subplot(3, 2, 2)
variantgq_SOR = variantqc_table["va.info.SOR"]
plt.hist(variantgq_SOR.dropna(), bins = np.arange(0, 8, 0.2))
plt.xlabel("SOR")
plt.ylabel("Frequency")
plt.xlim(0, 8)
plt.axvline(4, color = 'r')

plt.subplot(3, 2, 3)
variantgq_QD = variantqc_table["va.info.QD"]
plt.hist(variantgq_QD.dropna(), bins = np.arange(0, 40, 1))
plt.xlabel("QD")
plt.ylabel("Frequency")
plt.xlim(0, 40)
plt.axvline(2, color = 'r')

plt.subplot(3, 2, 4)
variantgq_FS = variantqc_table["va.info.FS"]
plt.hist(variantgq_FS.dropna(), bins = np.arange(0, 100, 2))
plt.xlabel("FS")
plt.ylabel("Frequency")
plt.xlim(0, 100)
plt.axvline(60, color = 'r')

plt.subplot(3, 2, 5)
variantgq_MQRankSum = variantqc_table["va.info.MQRankSum"]
plt.hist(variantgq_MQRankSum.dropna(), bins = np.arange(-20, 20, 1))
plt.xlabel("MQRankSum")
plt.ylabel("Frequency")
plt.xlim(-25, 25)
plt.axvline(-20, color = 'r')

plt.subplot(3, 2, 6)
variantgq_ReadPosRankSum = variantqc_table["va.info.ReadPosRankSum"]
plt.hist(variantgq_ReadPosRankSum.dropna(), bins = np.arange(-20, 20, 0.5))
plt.xlabel("ReadPosRankSum")
plt.ylabel("Frequency")
plt.xlim(-12, 12)
plt.axvline(-10, color = 'r')
plt.axvline(10, color = 'r')

plt.tight_layout()
plt.show()
display()

# COMMAND ----------

# MAGIC %md ## PCA
# MAGIC 
# MAGIC To check if there is any genetic structure, we will use a principal component analysis (PCA).

# COMMAND ----------

vds_pca = (vds_gDP_gGQ_vCR_sDP_sGT_vFilter.pca(scores='sa.pca'))

# COMMAND ----------

# MAGIC %md We can then make a Python plot of the samples in PC space colored by population group:

# COMMAND ----------

pca_table = vds_pca.samples_keytable().to_pandas()
colors = {'ASI': 'green', 'EUR': 'red', 'ME': 'blue'}
plt.clf()
plt.figure(figsize=(7,7)) # figure size in inches; change according to your screen size and resolution
plt.scatter(pca_table["sa.pca.PC1"], pca_table["sa.pca.PC2"], c = pca_table["sa.myAnnot.Population"].map(colors), alpha = .7, s=100)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.xlim(-0.7, 0.7)
plt.ylim(-0.7, 0.7)
legend_entries = [mpatches.Patch(color= c, label=myAnnot) for myAnnot, c in colors.items()]
plt.legend(handles=legend_entries, prop={'size': 15})
plt.show()
display()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC This plot reflects the known [population structure in *Capsella bursa-pastoris*](http://onlinelibrary.wiley.com/doi/10.1111/mec.13491/full).
# MAGIC 
# MAGIC You can see the location of these samples with some R code:

# COMMAND ----------

# MAGIC %md Install packages that are necessary to produce a geographic map:

# COMMAND ----------

# MAGIC %r 
# MAGIC install.packages(c("maps", "rworldmap"))

# COMMAND ----------

# MAGIC %r
# MAGIC library(rworldmap)
# MAGIC library(maps)
# MAGIC 
# MAGIC geo <- read.table("/databricks/driver/Cbp31_annot.csv", header = T) # read file not from dbfs
# MAGIC 
# MAGIC newmap <- getMap(resolution = "hight") # create a map object
# MAGIC 
# MAGIC pchS <- c(rep(15, 11), rep(19, 13), rep(17, 7)) # designate genetic clusters with different colours
# MAGIC colS <- c(rep("green", 11), rep("red", 13), rep("blue", 7))
# MAGIC groupsL <- c('ASI', 'EUR', 'ME')
# MAGIC pchL <- c(15, 19, 17)
# MAGIC colL <- c("green", "red", "blue")
# MAGIC 
# MAGIC par(mar=c(3, 3, 2, 2), cex=1)
# MAGIC plot(newmap, xlim = c(-40, 140), ylim = c(20, 50), asp = 1, bg='#DCDCDC', lty=3, lwd=0.3, col="#ffffff")
# MAGIC map.axes()
# MAGIC points(x=geo$Longitude, y=geo$Latitude, pch=pchS, cex=1.2, col=colS)
# MAGIC legend("topright", leg=groupsL, pch=pchL, col=colL, ncol=1, pt.cex=1.2, bg="#ffffff")

# COMMAND ----------

# MAGIC %md ## Summary
# MAGIC 
# MAGIC Data filtering:
# MAGIC  - Filter genotypes
# MAGIC  - Filter samples
# MAGIC  - Filter variants
# MAGIC 
# MAGIC *Variants can be filtered before samples filtering if samples are of greater priority in a study.*
# MAGIC 
# MAGIC Such genetic data can be analyzed in various ways. A PCA is just one simple example.

# COMMAND ----------

# MAGIC %md ## Additional
# MAGIC 
# MAGIC It is recommended to go through the original [Analysis of 1000 Genomes with Spark and Hail](https://docs.databricks.com/spark/latest/training/1000-genomes.html).
# MAGIC 
# MAGIC You can also read [Hail Overview](https://www.hail.is/hail/overview.html), look through the [Hail objects](https://www.hail.is/hail/hail_objects.html) representing many core concepts in genetics, and check out the many Hail functions defined in the [Python API](https://hail.is/hail/api.html).
