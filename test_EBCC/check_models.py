import json

with open("./demo_cerebellum.json", "r") as json_file:
    demo_cerebellum_config = json.load(json_file)
with open("./data/mouse_cerebellum_config_healthy.json", "r") as json_file:
    mouse_cerebellum_config_healthy = json.load(json_file)
#%%
check_neurons_models = False
demo_neuron_models = demo_cerebellum_config["cell_types"]
mouse_healthy_neuron_models = mouse_cerebellum_config_healthy["simulations"][
    "DCN_update"
]["cell_models"]
neuron_to_check = [
    "basket_cell",
    "dcn_cell_GABA",
    "dcn_cell_Gly-I",
    "dcn_cell_glut_large",
    "golgi_cell",
    "granule_cell",
    "purkinje_cell",
    "stellate_cell",
    "io_cell",
]
if check_neurons_models:

    for cell in neuron_to_check:
        print(cell)

        for property in mouse_healthy_neuron_models[cell]["parameters"].keys():
            assert (
                demo_neuron_models[cell]["parameters"][property]
                == mouse_healthy_neuron_models[cell]["parameters"][property]
            ), f"{cell}, {property}"
        if cell == "io_cell":
            for property in mouse_healthy_neuron_models[cell]["iaf_cond_alpha"].keys():
                if property != "receptors":
                    demo = demo_neuron_models[cell]["parameters"][property]
                    ali = mouse_healthy_neuron_models[cell]["iaf_cond_alpha"][property]
                    assert demo == ali, f"{cell}, {property}: demo={demo} ali={ali}"
        else:
            for property in mouse_healthy_neuron_models[cell][
                "eglif_cond_alpha_multisyn"
            ].keys():
                if property != "receptors":
                    demo = demo_neuron_models[cell]["parameters"][property]
                    ali = mouse_healthy_neuron_models[cell][
                        "eglif_cond_alpha_multisyn"
                    ][property]
                    assert demo == ali, f"{cell}, {property}: demo={demo} ali={ali}"
#%%
check_connections_model=True
if check_connections_model:
    demo_conn_models = demo_cerebellum_config["connection_models"]
    mouse_healthy_conn_models = mouse_cerebellum_config_healthy["simulations"]["DCN_update"]["connection_models"]
    conn_models_to_check = demo_conn_models.keys()
    for conn in conn_models_to_check:
        demo = [demo_conn_models[conn]["weight"], demo_conn_models[conn]["delay"]]
        ali = [mouse_healthy_conn_models[conn]["connection"]["weight"], mouse_healthy_conn_models[conn]["connection"]["delay"]]
        assert demo==ali, f"{conn}, weight: demo={demo[0]} ali={ali[0]}, delay: demo={demo[1]} ali={ali[1]}"