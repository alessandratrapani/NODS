#%%
import h5py
import dill


def save_dict_to_hdf5(file_path, data_dict):
    with h5py.File(file_path, 'w') as hdf_file:
        recursively_save_dict_contents(hdf_file, '/', data_dict)

def recursively_save_dict_contents(hdf_file, path, data_dict):
    for key, item in data_dict.items():
        if key!="param_vt" and key!="vt_num":
            if isinstance(item, (dict)):
                recursively_save_dict_contents(hdf_file, path + key + '/', item)
            else:
                hdf_file[path + key] = item


data_path = "./data/"
hdf5_file = "cerebellum_300x_200z.hdf5"
scaffold = h5py.File(data_path + hdf5_file, "r")
neuronal_populations_from_hdf5 = scaffold["cells"]["placement"]
connectivity_from_hdf5 = scaffold["cells"]["connections"]

#%%
create_new = False
if create_new:
    network_geom_file = data_path + "geom_" + hdf5_file
    network_connectivity_file = data_path + "conn_" + hdf5_file
    neuronal_populations = dill.load(open(network_geom_file, "rb"))
    connectivity = dill.load(open(network_connectivity_file, "rb"))

    save_dict_to_hdf5(data_path + "new_geom_" + hdf5_file, neuronal_populations)
    save_dict_to_hdf5(data_path + "new_conn_" + hdf5_file, connectivity)
#%%
check_connections=False
if check_connections:
    neuronal_populations = h5py.File(data_path + "new_geom_" + hdf5_file, "r")
    connectivity = h5py.File(data_path + "new_conn_" + hdf5_file, "r")

    for conn in connectivity:
        if not conn=="io_to_vt":
            print(conn)
            id_pre = connectivity[conn]["id_pre"]
            id_post = connectivity[conn]["id_post"]
            differences = []
            for i in range(len(connectivity_from_hdf5[conn])):
                if id_pre[i] != int(connectivity_from_hdf5[conn][i][0])+1:
                    differences.append([0,i])
                if id_post[i] != int(connectivity_from_hdf5[conn][i][1])+1:
                    differences.append([1,i])
            print(differences)
#%%