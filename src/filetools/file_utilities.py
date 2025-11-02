import csv
import os
from typing import Any, Callable, Iterable
import constants as c

def read_csv(
    csv_filepath:str,
    include_headers:bool,
    callback_fn_on_row:Callable[[Any],Any]
)->list[Any]:
    results = []
    with open(csv_filepath, newline='') as csvfile:
        #rows = csv.reader(csvfile, delimiter=' ', quotechar='|')
        rows = csv.reader(csvfile)
        for counter,row in enumerate(rows):
            if counter > 0 or (counter==0 and include_headers):
                res = callback_fn_on_row(row)
                results.append(res)
    return results

def save_list_as_csv(
    filename:str,
    data:list[list]
)->None:
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)        
        # Write the data
        writer.writerows(data)


def append_dictionary_to_csv(
        csv_filepath:str,
        row:dict[str,Any],
        write_header:bool=False
    )->None:    
   
    # field names
    fields = list(row.keys())
    
    if write_header:
        file_mode="w" # erase file and add a row
    else:
        file_mode="a" #append rows

    # writing to csv file
    with open(csv_filepath, file_mode, encoding="utf-8") as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        if write_header:
            # writing headers (field names)
            writer.writeheader()

        # writing a row
        writer.writerow(row)
        csvfile.close()

def read_csv_all_rows(
    csv_filepath:str,
    include_headers:bool=True
)->list[list[str]]:
    def callback_fn_on_row(row):
        return row
    return read_csv(
        csv_filepath=csv_filepath,
        include_headers=include_headers,
        callback_fn_on_row=callback_fn_on_row
    )

def dictionary_lists_to_csv(
        csv_filepath:str,
        rows:list[dict[str,Any]]
    )->None:    
   
    # field names
    fields = list(rows[0].keys())

    # writing to csv file
    with open(csv_filepath, 'w', encoding="utf-8") as csvfile:
        # creating a csv dict writer object
        writer = csv.DictWriter(csvfile, fieldnames=fields)

        # writing headers (field names)
        writer.writeheader()

        # writing data rows
        writer.writerows(rows)

        csvfile.close()


def iterate_files_in_dir(
        dir_info:dict[str,Any],
        callback_fn_on_file:Callable[[dict[str,Any],str,int],Any]
    )->list[Any]:

    dir = dir_info["dir"]                    
    print("Will process files in directory:", dir)
    total_files = len(os.listdir(dir))
    print("total files to process:", total_files)
    
    file_indices = None
    if c.DIRINFO_KEYNAME_PROCESS_ONLY_FILE_INDICES in dir_info:
        file_indices=dir_info[c.DIRINFO_KEYNAME_PROCESS_ONLY_FILE_INDICES]

    results = []
    for count,filename in enumerate(os.listdir(dir)):
        file_index = count+1
        print("\rprocessing file index:", file_index, ", filename:", filename, ", total files: ", total_files, "                   ",end="")            
        if file_indices == None or file_index in file_indices:                        
            result = callback_fn_on_file(dir_info, filename, file_index)            
        
        
        #if result is not None:            
        #    results.append(result)        
    print("")
    return results


def iterate_files_in_directories(
        directories_info:list[dict[str,Any]],
        callback_fn_on_file:Callable[[dict[str,Any],str,int],Any],
        callback_fn_on_dir_processed:Callable[[dict[str,Any],list[Any]],None]        
    )->None:

    for dir_info in directories_info:
        results = iterate_files_in_dir(dir_info, callback_fn_on_file)        
        callback_fn_on_dir_processed(dir_info,results)


def get_cvus_from_gexf_dirs(
    directories_info:list[dict[str,Any]]
)->list[str]:
    cvus:list[str] = []
    
    def callback_fn_on_file(        
        dir_info: dict[str,Any],
        filename:str,
        file_number:int
    )->Any:
        cvu = filename.replace(c.CVU_PREFFIX,"").replace(c.GEXF_EXTENSION,"")
        cvus.append(cvu)
        return cvu
    
    def callback_fn_on_dir_processed(
            directories_info:dict[str,Any],
            results:list[Any]
    ):
        pass

    iterate_files_in_directories(
        directories_info=directories_info,
        callback_fn_on_file=callback_fn_on_file,
        callback_fn_on_dir_processed=callback_fn_on_dir_processed
    )

    return cvus


