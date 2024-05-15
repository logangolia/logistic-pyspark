print('hi')

import re
import numpy as np


# load up all of the 19997 documents in the corpus
corpus = sc.textFile ("s3://chrisjermainebucket/comp330_A5/TrainingDataOneLinePerDoc.txt")
# corpus = sc.textFile ("s3://chrisjermainebucket/comp330_A5/TestingDataOneLinePerDoc.txt")


# each entry in validLines will be a line from the text file
validLines = corpus.filter(lambda x : 'id' in x)

# now we transform it into a bunch of (docID, text) pairs
keyAndText = validLines.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], x[x.index('">') + 2:]))

# now we split the text in each (docID, text) pair into a list of words
# after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# we have a bit of fancy regular expression stuff here to make sure that we do not
# die on some of the documents
regex = re.compile('[^a-zA-Z]')
keyAndListOfWords = keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))

# now get the top 20,000 words... first change (docID, ["word1", "word2", "word3", ...])
# to ("word1", 1) ("word2", 1)...
allWords = keyAndListOfWords.flatMap(lambda x: ((j, 1) for j in x[1]))

# now, count all of the words, giving us ("word1", 1433), ("word2", 3423423), etc.
allCounts = allWords.reduceByKey (lambda a, b: a + b)


# Sort by count descending, then alphabetically for ties
sortedWords = allCounts.sortBy(lambda x: (-x[1], x[0]))


# and get the top 20,000 words in a local array
# each entry is a ("word1", count) pair
topWords = sortedWords.top (20000, lambda x : x[1])

# and we'll create a RDD that has a bunch of (word, dictNum) pairs
# start by creating an RDD that has the number 0 thru 20000
# 20000 is the number of words that will be in our dictionary
twentyK = sc.parallelize(range(20000))

# now, we transform (0), (1), (2), ... to ("mostcommonword", 1) ("nextmostcommon", 2), ...
# the number will be the spot in the dictionary used to tell us where the word is located
# HINT: make use of topWords in the lambda that you supply
# dictionary = twentyK.map (?q?????????????????????)
# fetch the i-th most frequent word from topWords and assigns it the rank i
dictionary = twentyK.map (lambda num : (topWords[num][0], num)).cache()


# finally, print out some of the dictionary, just for debugging
dictionary.top (10)


# Words to look up in the dictionary
words_to_lookup = ["applicant", "and", "attack", "protein", "car"]

# Perform the lookup for each word and print the index or -1 if not found
for word in words_to_lookup:
    lookup_result = dictionary.lookup(word)
    if lookup_result:  # If the list is not empty, word is in the dictionary
        print(f"The word '{word}' is in the dictionary with an index of {lookup_result[0]}.")
    else:  # If the word is not found, return -1
        print(f"The word '{word}' is not in the dictionary, returning -1.")

# Task 2
switch_dict = dictionary.map(lambda x : (x[1], x[0]))

# Generate RDD with pairs of (word, document ID)
wordDocPairs = keyAndListOfWords.flatMap(lambda x: ((word, x[0]) for word in x[1]))

# Filter to include only words that are in the dictionary; pairs are (word, (document ID, index))
filteredWordDocs = wordDocPairs.join(dictionary)

# Transform to pairs of (document ID, index)
docIndexPairs = filteredWordDocs.map(lambda x: (x[1][0], x[1][1]))

# Count occurrences of each dictionary index within each document and group into pairs ((document ID, index), count)
indexCounts = docIndexPairs.map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)

# Convert to pairs (document ID, (index, count))
docIndexCountPairs = indexCounts.map(lambda x: (x[0][0], (x[0][1], x[1])))

# Group by document ID
groupedIndexCounts = docIndexCountPairs.groupByKey().mapValues(list)

# Convert grouped values into numpy arrays
def counts_to_array(counts):
    array = np.zeros(20000)
    for index, count in counts:
        array[index] = count
    return array

# Map document IDs to their word count arrays
docCountArrays = groupedIndexCounts.map(lambda x: (x[0], counts_to_array(x[1])))

# Calculate Term Frequency (TF) for each document
termFrequency = docCountArrays.map(lambda x: (x[0], x[1] / x[1].sum()))

# Create binary incidence matrix indicating the presence of words
binaryIncidence = docCountArrays.map(lambda x: np.where(x[1] > 0, 1, 0))

# Count documents to calculate Inverse Document Frequency (IDF)
totalDocuments = docCountArrays.count()

# Compute IDF using the binary incidence matrix
inverseDocFrequency = np.log(totalDocuments / binaryIncidence.reduce(lambda a, b: a + b))

# Calculate TF-IDF for each document
TF_IDF = termFrequency.map(lambda x: (x[0], np.multiply(x[1], inverseDocFrequency)))

# Display the first 10 entries of the TF-IDF RDD
TF_IDF.take(10)



# Calculate the mean TF-IDF vector across all documents
averageTF_IDF = TF_IDF.map(lambda x: x[1]).reduce(lambda a, b: a + b)
averageTF_IDF /= numDocs

# Calculate the standard deviation of the TF-IDF vectors
stdDevTF_IDF = TF_IDF.map(lambda x: (x[1] - averageTF_IDF) ** 2).reduce(lambda a, b: a + b)
stdDevTF_IDF = np.sqrt(stdDevTF_IDF / numDocs)

# Output average and standard deviation for TF-IDF vectors
print(averageTF_IDF)
print(stdDevTF_IDF)

# Normalize the TF-IDF vectors to have zero mean and unit variance
normalizedVectors = TF_IDF.map(lambda x: (x[0], np.nan_to_num((x[1] - averageTF_IDF) / stdDevTF_IDF)))
print(normalizedVectors.top(5))

# Label normalized vectors based on whether their document ID starts with 'AU'
labeledData = normalizedVectors.map(lambda x: (x[0], x[1], int('AU' == x[0][0:2]))).cache()




# Initialize the weights array
weights = np.zeros(20000)
# Learning rate
learning_rate = 1
# Iteration counter
iteration_count = 1
# Regularization parameter
regularization_strength = 2

# Calculate the initial value of the objective function
initial_obj = labeledData.map(
    lambda record: (-record[2] * np.dot(record[1], weights) + np.log(1 + np.exp(np.dot(record[1], weights))))
).reduce(lambda acc, curr: acc + curr)
initial_obj -= regularization_strength * np.linalg.norm(weights, 2) ** 2
initial_obj /= numDocs
max_iterations = 50

while iteration_count < max_iterations:
    # Calculate the gradient of the objective function
    gradient = labeledData.map(
        lambda record: ((-record[2] + np.exp(np.dot(record[1], weights)) / (1 + np.exp(np.dot(record[1], weights)))) * record[1])
    ).reduce(lambda acc, curr: acc + curr) - 2 * regularization_strength * weights
    gradient /= numDocs
    weights -= learning_rate * gradient
    
    # Recalculate the objective function to check for convergence
    current_obj = labeledData.map(
        lambda record: (-record[2] * np.dot(record[1], weights) + np.log(1 + np.exp(np.dot(record[1], weights))))
    ).reduce(lambda acc, curr: acc + curr)
    current_obj -= regularization_strength * np.linalg.norm(weights, 2) ** 2
    current_obj /= numDocs
    print(f'Objective after iteration {iteration_count} is {current_obj}')

    # Check for convergence
    if abs(current_obj - initial_obj) < 1e-5:
        print(f'Converged weights = {weights}')
        break
    # Adjust learning rate using bold driver strategy
    elif current_obj > initial_obj:
        learning_rate /= 2
    elif current_obj < initial_obj:
        learning_rate *= 1.1
    iteration_count += 1
    initial_obj = current_obj



# Initialize list to store the top 50 words with their coefficients
significantWords = []
# Select the indices of the top 50 highest coefficients
indicesTop50 = np.argpartition(weights, -50)[-50:]
# Retrieve the words corresponding to these indices and their coefficients
for index in indicesTop50:
    significantWords.append((switch_dict.lookup(index), weights[index]))
# Sort the list based on the coefficients in descending order
significantWords.sort(key=lambda pair: -pair[1])
# Output the words and their corresponding coefficients
for word, value in significantWords:
    print(word, value)



# TASK 3

# load up all of the documents in the corpus
#test_corpus = sc.textFile("s3://chrisjermainebucket/comp330_A5/SmallTrainingDataOneLinePerDoc.txt")
test_corpus = sc.textFile ("s3://chrisjermainebucket/comp330_A5/TestingDataOneLinePerDoc.txt")

# each entry in validLines will be a line from the text file
testValidLines = test_corpus.filter(lambda x: 'id' in x)

# now we transform it into a bunch of (docID, text) pairs
testKeyAndText = testValidLines.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], x[x.index('">') + 2:]))

# # now we split the text in each (docID, text) pair into a list of words
# # # after this, we have a data set with (docID, ["word1", "word2", "word3", ...])
# # # we have a bit of fancy regular expression stuff here to make sure that we do not
# # # die on some of the documents
testRegex = re.compile('[^a-zA-Z]')
testKeyAndListOfWords = testKeyAndText.map(lambda x: (str(x[0]), testRegex.sub(' ', x[1]).lower().split()))

# Flatten the list of words into pairs of words and document IDs
testWordInDocs = testKeyAndListOfWords.flatMap(lambda x: ((j, x[0]) for j in x[1]))

# Join with the dictionary to filter out words not in the top 20000, resulting in pairs of form (word, (docID, index))
testDictWords = testWordInDocs.join(dictionary)

# Map to pairs of document IDs and dictionary indices
testDocIDWithIdx = testDictWords.map(lambda x: (x[1][0], x[1][1]))

# Count occurrences of each dictionary index within documents
testWordCountsByDoc = testDocIDWithIdx.map(lambda x: (x, 1)).reduceByKey(lambda a, b: a + b)

# Convert to pairs of document IDs and tuples of indices and counts
testDocIDWithCounts = testWordCountsByDoc.map(lambda x: (x[0][0], (x[0][1], x[1])))

# Group by document ID to collect all index-count pairs per document
testGroupedCounts = testDocIDWithCounts.groupByKey().mapValues(list)

# Convert grouped counts into numpy arrays representing the document vectors
def convertToNumpyArray(counts):
    arr = np.zeros(20000)
    for idx, count in counts:
        arr[idx] = count
    return arr

testDocVectors = testGroupedCounts.map(lambda x: (x[0], convertToNumpyArray(x[1])))

# Calculate term frequencies by normalizing document vectors
testTermFrequencies = testDocVectors.map(lambda x: (x[0], x[1] / np.sum(x[1]) if np.sum(x[1]) != 0 else np.zeros(20000)))

# Generate incidence matrix to calculate IDF
testIncidenceMatrix = testDocVectors.map(lambda x: np.where(x[1] > 0, 1, 0))

# Calculate the total number of documents
testNumDocs = testDocVectors.count()

# Compute IDF using the incidence matrix
testInverseDocFreq = np.log(testNumDocs / testIncidenceMatrix.reduce(lambda a, b: a + b))

# Compute TF-IDF vectors
testTF_IDF = testTermFrequencies.map(lambda x: (x[0], np.multiply(x[1], testInverseDocFreq)))

# Normalize the test TF-IDF vectors using the mean and standard deviation from the training data
normalizedTestTF_IDF = testTF_IDF.map(
    lambda x: (x[0], np.nan_to_num((x[1] - train_mean) / train_stddev))
)
# Display the top 5 normalized TF-IDF vectors to verify the transformation
print(normalizedTestTF_IDF.top(5))

# Label the normalized test TF-IDF vectors based on whether the document ID starts with 'AU'
# This step assigns a binary label for a classification task, where '1' indicates Australian cases
labeledNormalizedTestTF_IDF = normalizedTestTF_IDF.map(
    lambda x: (x[0], x[1], int('AU' == x[0][0:2]))
).cache()  # Cache the RDD for efficient access during iterative processes



# Generate predictions and actual labels along with document IDs

document_predictions = labeledNormalizedTestTF_IDF.map(lambda x: (x[0], np.dot(x[1], weights), x[2]))
print(document_predictions.top(20))

# Function to categorize predictions based on a threshold
def classify_predictions(data, cutoff):
    doc_id, predicted_score, actual_label = data
    if (predicted_score > cutoff) and (actual_label == 1):
        return 0  # True Positive
    elif (predicted_score < cutoff) and (actual_label == 0):
        return 1  # True Negative
    elif (predicted_score > cutoff) and (actual_label == 0):
        return int(doc_id)  # False Positive (return document ID)
    else:
        return 3  # False Negative

# Apply classification to predictions and count occurrences of each category
prediction_outcomes = document_predictions.map(lambda x: classify_predictions(x, 20)).countByValue()

# Calculate correct predictions and metrics for evaluation
correct_predictions = prediction_outcomes[0] + prediction_outcomes[1]
total_true_positives = prediction_outcomes[0] + prediction_outcomes[3]
total_predicted_positives = prediction_outcomes[0]

# Sum up all false positives
total_false_positives = sum(count for key, count in prediction_outcomes.items() if key not in [0, 1, 3])

print(f'Total false positives: {total_false_positives}')
print(f'{correct_predictions} out of {testNumDocs} correct.')
print(f'Total actual positives: {total_true_positives}, Total predicted as positives: {total_predicted_positives + total_false_positives}, '
      f'F1 Score: {2 * total_predicted_positives / (total_predicted_positives + total_false_positives + total_true_positives) if (total_predicted_positives + total_false_positives + total_true_positives) > 0 else 0}')


count = 0
for k in prediction_outcomes.keys():
    if k != 0 and k != 1 and k != 3:
        count += 1
        print(k, testKeyAndText.lookup(str(k)), '\n')
print(count)




