import constants as c
########################### GXFILE DIRECTORIES #######################
CVU_GXFILE_DIRS_INFO_EGO_INCLUDED = []
CVU_GXFILE_DIRS_INFO_EGO_REMOVED = []

__GXFILE_DIR = "/Users/irobles/Documents/workspaces/python/SNI/sni-personal-nets/networks"
__CSV_DIR = "/Users/irobles/Documents/workspaces/python/SNI/sni-net-builders/data"

__HEALTH_INCLUDE_EGO_CSV_FILE_NAME = "salud_ego_included_coco.csv"
__ENGINEERS_INCLUDE_EGO_CSV_FILE_NAME = "ingenieros_ego_included_coco.csv"
__ECONOMISTS_INCLUDE_EGO_CSV_FILE_NAME = "economistas_ego_included_coco.csv"

__HEALTH_REMOVED_EGO_CSV_FILE_NAME = "salud_ego_removed_coco.csv"
__ENGINEERS_REMOVED_EGO_CSV_FILE_NAME = "ingenieros_ego_removed_coco.csv"
__ECONOMISTS_REMOVED_EGO_CSV_FILE_NAME = "economistas_ego_removed_coco.csv"

###### INCLUDE EGO ##########
__health_dir_info = dict()
__health_dir_info[c.DIRINFO_KEYNAME_DIR] = __GXFILE_DIR + "/salud/"
__health_dir_info[c.DIRINFO_KEYNAME_CSV_FILEPATH] = __CSV_DIR + "/" + __HEALTH_INCLUDE_EGO_CSV_FILE_NAME
#__health_dir_info[c.DIRINFO_KEYNAME_PROCESS_ONLY_FILE_INDICES] = [702]
CVU_GXFILE_DIRS_INFO_EGO_INCLUDED.append(__health_dir_info)

__engineers_dir_info = dict()
__engineers_dir_info[c.DIRINFO_KEYNAME_DIR] = __GXFILE_DIR + "/ingenieros/"
__engineers_dir_info[c.DIRINFO_KEYNAME_CSV_FILEPATH] = __CSV_DIR + "/" + __ENGINEERS_INCLUDE_EGO_CSV_FILE_NAME
CVU_GXFILE_DIRS_INFO_EGO_INCLUDED.append(__engineers_dir_info)

__economists_dir_info = dict()
__economists_dir_info[c.DIRINFO_KEYNAME_DIR] = __GXFILE_DIR + "/economistas/"
__economists_dir_info[c.DIRINFO_KEYNAME_CSV_FILEPATH] = __CSV_DIR + "/" + __ECONOMISTS_INCLUDE_EGO_CSV_FILE_NAME
CVU_GXFILE_DIRS_INFO_EGO_INCLUDED.append(__economists_dir_info)

################################## REMOVE EGO ################################
__health_dir_info = dict()
__health_dir_info[c.DIRINFO_KEYNAME_DIR] = __GXFILE_DIR + "/salud/"
__health_dir_info[c.DIRINFO_KEYNAME_CSV_FILEPATH] = __CSV_DIR + "/" + __HEALTH_REMOVED_EGO_CSV_FILE_NAME
#__health_dir_info[c.DIRINFO_KEYNAME_PROCESS_ONLY_FILE_INDICES] = [4]
CVU_GXFILE_DIRS_INFO_EGO_REMOVED.append(__health_dir_info)

__engineers_dir_info = dict()
__engineers_dir_info[c.DIRINFO_KEYNAME_DIR] = __GXFILE_DIR + "/ingenieros/"
__engineers_dir_info[c.DIRINFO_KEYNAME_CSV_FILEPATH] = __CSV_DIR + "/" + __ENGINEERS_REMOVED_EGO_CSV_FILE_NAME
#__engineers_dir_info[c.DIRINFO_KEYNAME_PROCESS_ONLY_FILE_INDICES] = [8]
CVU_GXFILE_DIRS_INFO_EGO_REMOVED.append(__engineers_dir_info)

__economists_dir_info = dict()
__economists_dir_info[c.DIRINFO_KEYNAME_DIR] = __GXFILE_DIR + "/economistas/"
__economists_dir_info[c.DIRINFO_KEYNAME_CSV_FILEPATH] = __CSV_DIR + "/" + __ECONOMISTS_REMOVED_EGO_CSV_FILE_NAME
#__economists_dir_info[c.DIRINFO_KEYNAME_PROCESS_ONLY_FILE_INDICES] = [12]
CVU_GXFILE_DIRS_INFO_EGO_REMOVED.append(__economists_dir_info)