import jpype
import jpype.imports
from jpype.types import *

moa_jar_path = "moa/moa.jar"
weka_jar_path = "moa/weka-dev-3.9.2.jar"

# Step 2: Start the JPype JVM with the MOA and Weka JARs
jpype.startJVM(classpath=[moa_jar_path, weka_jar_path])

# Step 3: Import MOA classes
from moa.streams.clustering import FileStream
from moa.clusterers.clustree import ClusTree
from moa.evaluation import CMM

# Step 4: Set up the FileStream (data source)
data_file_path = "data/kddcup/kddcup.arff"  # Replace with your ARFF file path
stream = FileStream()
stream.arffFileOption.setValue(data_file_path)
stream.classIndexOption.setValue(-1)
stream.prepareForUse()

# Step 5: Initialize ClusTree clustering model
clustree = ClusTree()
clustree.prepareForUse()

# Step 6: Process the FileStream and train ClusTree
num_instances_to_process = 1000  # Number of instances to process
count = 0
while stream.hasMoreInstances() and count < num_instances_to_process:
    instance_example = stream.nextInstance()
    instance = instance_example.getData()  # Get the instance data
    
    # Train the clustering model with the instance
    clustree.trainOnInstanceImpl(instance)
    count += 1

# Step 7: Evaluate the clustering using CMM
# Retrieve the clustering result (micro-clusters)
clustering_result = clustree.getClusteringResult()

# Create a CMM instance and set the ground truth and clustering result
cmm = CMM()
cmm.setClustering(clustering_result)  # Set the clustering result from ClusTree
cmm.setGroundTruth(stream.getHeader())  # Ground truth is provided by the FileStream

# Calculate the CMM metrics
precision = cmm.getPrecisionStatistic()
recall = cmm.getRecallStatistic()
f_measure = cmm.getFStatistic()

# Display results
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F-Measure: {f_measure}")

# Step 8: Stop the JVM
jpype.shutdownJVM()