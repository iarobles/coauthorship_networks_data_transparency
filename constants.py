
########################### CATEGORIES #######################
CATEGORY_GENDER_FEMALE = "f"
CATEGORY_GENDER_MALE = "m"
CATEGORY_GENDER_UNKNOWN = "u"


########################### DIR INFO KEYNAMES #######################
DIRINFO_KEYNAME_CSV_FILEPATH = "csv_metrics_filepath"
DIRINFO_KEYNAME_DIR = "dir"
DIRINFO_KEYNAME_PROCESS_ONLY_FILE_INDICES = "process_only_file_indices"


########################### METRIC KEYNAMES #######################

# NODE METRICS:
# basic metrics:
KEYNAME_HINDEX = "hindex"
KEYNAME_CITES_COUNT = "cites_count"
KEYNAME_PUBLICATIONS_COUNT = "publications_count"
# centralities:
KEYNAME_DEGREE_CENTRALITY = "degree_centrality"
KEYNAME_CLOSENESS_CENTRALITY = "closeness_centrality"
KEYNAME_CLUSTERING_COEFFICIENT = "clustering_coefficient"
KEYNAME_BETWEENESS_CENTRALITY = "betweenness_centrality"
# K core:
# see: https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.core.core_number.html#networkx.algorithms.core.core_number
KEYNAME_CORE_NUMBER = "core_number"
# STRUCTURAL HOLES
# See: https://networkx.org/documentation/stable/reference/algorithms/structuralholes.html
KEYNAME_STRUCTURAL_HOLES_CONSTRAINT = "structural_holes_constraint"
KEYNAME_STRUCTURAL_HOLES_EFFECTIVE_SIZE = "structural_holes_effective_size"

# GRAPH METRICS
KEYNAME_AVG_DEGREE = "average_degree"
KEYNAME_AVG_CLUSTERING = "average_clustering"
KEYNAME_AVG_NODE_CONNECTIVITY = "average_node_connectivity"
KEYNAME_DIAMETER = "diameter"
KEYNAME_DENSITY = "density"
KEYNAME_DOMINATION_NUMBER = "domination_number"
KEYNAME_CHROMATIC_NUMBER = "chromatic_number"
KEYNAME_NUMBER_CONNECTED_COMPONENTS = "number_connected_components"
KEYNAME_DEGREE_ASSORTATIVITY = "degree_assortativity"
KEYNAME_HINDEX_ASSORTATIVITY = "hindex_assortativity"
KEYNAME_CITES_ASSORTATIVITY = "cites_assortativity"
KEYNAME_PUBLICATIONS_ASSORTATIVITY = "publications_assortativity"
#KEYNAME_ASSORTATIVITY = "Assortativity"

#"Closeness_origin"
#'Closeness_residence'
#'Number_origin','Number_residence','Mu'

########################### GRAPH KEYNAMES #######################
KEYNAME_GRAPH_ID = "graphid"
KEYNAME_NODE_ATTRIBUTE_HINDEX = "hindex"
KEYNAME_NODE_ATTRIBUTE_WEIGHT = "weight"
KEYNAME_NODE_ATTRIBUTE_PUBLICATIONS_COUNT = "pub_count"
KEYNAME_NODE_ATTRIBUTE_CITES_COUNT = "cites_count"

########################### GENERIC QUERY RESULTS KEYNAMES #######################
KEYNAME_QUERY_RESULT_EXISTS = "exists"
KEYNAME_CVU = "cvu"
KEYNAME_PAPERS_CITE_COUNT = "papers_cite_count"
KEYNAME_HINDEX = "hindex"
KEYNAME_AUTHOR_SCOPUS_ID = "author_scopus_id"
KEYNAME_PUBLICATIONS_COUNT = "publications_count"
KEYNAME_CITES_COUNT = "cites_count"


########################### FILE EXTENSIONS #######################
GEXF_EXTENSION = ".gexf"


########################### PREFFIXES #######################
CVU_PREFFIX = "cvu_"
SCOPUS_ID_PREFFIX = "sid_"

