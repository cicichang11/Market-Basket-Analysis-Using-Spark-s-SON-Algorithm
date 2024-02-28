from pyspark import SparkContext
import sys
import time
from operator import add
from collections import defaultdict
from collections import defaultdict, Counter
from itertools import combinations


# preprocess the input data based on the case number
def prepare_data(row, case_number):
    if case_number not in (1, 2):
        raise ValueError("Case number must be either 1 or 2")
    # split each row of the csv data 
    user_id, business_id = row.split(',')
    if case_number == 1:
        return (user_id, business_id)
    else:
        return (business_id, user_id)
    

# find frequent singletons
def find_freq_singletons(item_baskets, support):
    # dictionary to count each unique item's occurrences
    item_counts = defaultdict(int)
    item_counts = Counter(item for basket in item_baskets for item in basket)
    
    # a set to hold items that meet or exceed the support threshold
    frequent_singles = {frozenset([item]) for item, count in item_counts.items() if count >= support}
    
    return frequent_singles



# find one-level-up frequents
def one_level_up_candidates(previousF, size):
    # a set to store the new candidates
    new_candidates = set()
    # iterate over all pairs of previous frequent itemsets
    for item1 in previousF:
        for item2 in previousF:
            # create a new candidate by taking the union of two itemsets
            candidate = item1 | item2  
            # ensure the new candidate has the desired size
            if len(candidate) == size:
                # check if all subsets of the new candidate are frequent
                if all(frozenset(subset) in previousF for subset in combinations(candidate, size - 1)):
                    # add the new candidate as a frozenset to ensure immutability
                    new_candidates.add(frozenset(candidate))
    return new_candidates


# from the candidates generate find the frequent ones
def freq_candidates(item_baskets, candidates, support):
    # initialize a dictionary to track counts of potential itemsets
    itemset_counts = defaultdict(int)
    # iterate over each potential itemset
    for itemset in candidates:
        # check each basket to see if it contains the itemset
        for basket in item_baskets:
            # if the entire itemset is present in the basket, increment its count
            if itemset.issubset(basket):
                itemset_counts[frozenset(itemset)] += 1
    # Filter out itemsets that meet or exceed the minimum support threshold
    frequent_itemsets = {itemset for itemset, count in itemset_counts.items() if count >= support}

    return frequent_itemsets



# get frequent items using the A-Priori algorithm
def a_priori(item_baskets, support):
    # initialize lists to hold frequent itemsets and the last set of frequent itemsets found
    allFreq, previousF = [], []
    iteration = 1

    # continuously search for frequent itemsets of increasing size until none are found
    while True:
        # handle the base case for singletons separately
        if iteration == 1:
            current_frequents = find_freq_singletons(item_baskets, support)
        else:
            # generate candidate itemsets based on the itemsets found in the previous iteration
            candidates = one_level_up_candidates(previousF, iteration)
            # find the frequent itemsets among them
            current_frequents = freq_candidates(item_baskets, candidates, support)

            # Exit the loop if no frequent itemsets are found in this iteration
            if not current_frequents:
                break

        # store the frequent itemsets found in this iteration
        allFreq.append(current_frequents)

        # prepare for the next iteration
        iteration += 1
        previousF = current_frequents
    
    return allFreq



'''You should use “Candidates:” as the tag. For each line you should output the candidates of frequent
itemsets you found after the first pass of SON Algorithm followed by an empty line after each
combination.'''
def consolidate_candidates(frequent_sets):
    # initialize a set to hold all unique candidates
    unique_candidates = set()

    # iterate over each set of frequent itemsets and add them to the set of unique candidates
    for frequent_items in frequent_sets:
        unique_candidates.update(frequent_items)
    
    # convert the set of unique candidates back to a list for consistency with the original function's return type
    aggregated_candidates = list(unique_candidates)
    
    return aggregated_candidates



# save output file
def save_frequent_itemsets(output_file_path, frequent_itemsets):
    # sort itemsets first by length, then lexicographically
    sorted_itemsets = sorted([sorted(itemset) for itemset in frequent_itemsets], key=lambda x: (len(x), x))
    # append mode: any data written to the file is automatically added to the end
    with open(output_file_path, "a") as output_file:
        # arack the current itemset size for formatting purposes
        current_size = 0
        items_to_write = []

        for itemset in sorted_itemsets:
            if len(itemset) != current_size:
                # when the itemset size changes, write the accumulated itemsets to the file and reset the list
                if items_to_write:
                    output_file.write(', '.join(items_to_write) + "\n\n")
                    items_to_write = []
                current_size = len(itemset)

            # format the current itemset as a string and add it to the list
            formatted_itemset = "(" + ", ".join(f"'{item}'" for item in itemset) + ")"
            items_to_write.append(formatted_itemset)

        # write any remaining itemsets to the file
        if items_to_write:
            output_file.write(', '.join(items_to_write) + "\n\n")



# execution function
def execute(sc, input_file_path, output_file_path, case_number, support):
    rdd = sc.textFile(input_file_path)
    header = rdd.first()
    # rdd = rdd.filter(lambda x: x != header).map(lambda x: x.split(","))
    rdd = rdd.filter(lambda x: x != header).map(lambda row: prepare_data(row, case_number))

    if case_number == 1 or case_number == 2:
        baskets = rdd.map(lambda x: (x[0], x[1])).groupByKey().mapValues(set)
    else:
        raise ValueError("Invalid case number. Case number must be either 1 or 2.")
    # create backets 
    baskets = baskets.map(lambda x: x[1])
    num_basket = baskets.count()

    # SON phase 1 map
    def son_phase1_map(iterator):
        item_baskets = list(iterator)
        local_support = len(item_baskets) / num_basket * support
        return a_priori(item_baskets, local_support) 
    local_freq_list = baskets.mapPartitions(son_phase1_map).collect()
    # SON phase 1 reduce 
    candidates = consolidate_candidates(local_freq_list)


    # SON phase 2 map
    def son_phase2_map(iterator):
        item_baskets = list(iterator)
        occur = defaultdict(int)
        for candidate in candidates:
            for basket in item_baskets:
                if candidate.issubset(basket):
                    occur[frozenset(candidate)] += 1
        return [(key, occur[key]) for key in occur.keys()]
    local_counts = baskets.mapPartitions(son_phase2_map)
    # SON phase 2 reduce 
    frequents = local_counts.reduceByKey(add).filter(lambda x: x[1] >= support).map(lambda x: set(x[0]))
    result = frequents.collect()

    
    # write to output files
    with open(output_file_path, "w") as outFile:
        outFile.write("Candidates:\n")
    save_frequent_itemsets(output_file_path, candidates)

    with open(output_file_path, "a") as outFile:
        outFile.write("Frequent Itemsets:\n")
    save_frequent_itemsets(output_file_path, result)




if __name__ == "__main__":
    try:
        # count start time
        start_time = time.time()
        
        # load files 
        case_number = int(sys.argv[1])
        support = int(sys.argv[2])
        input_file_path = sys.argv[3]
        output_file_path = sys.argv[4]
       
       # initiate sparkcontext
        sc = SparkContext.getOrCreate()
        
        execute(sc, input_file_path, output_file_path, case_number, support)

        # execution time
        end_time = time.time()
        print("Duration:", end_time - start_time)

    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
