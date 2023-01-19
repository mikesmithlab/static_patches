from conditions import sim_params
from physics import Engine
from reader import Animator
from analyser import plot_energy, plot_patches, plot_charges, show_plots

# todo
# air resistance? linear and/or rotational ---- do calculations
# tiny tiny bit of randomness in the collision forces (more realistic)?
# measure all physical parameters:
# amplitude
# container_radius
# density
# coefficient_of_restitution
# gamma_t (viscous damping coefficient)
# mu (coefficient of friction)
# todo optimise for speed (side goal)

# this link laughs in my face
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html



def main():
  
    """
    # do_analysis = False
    analyse = False
    if analyse:
        print("analysing....")
        # do_energy_analysis = False
        do_energy_analysis = True
        do_patch_analysis = False
        # do_patch_analysis = True
        # do_charge_analysis = False
        do_charge_analysis = True
        if do_energy_analysis:
            plot_energy(conds["time_end"], conds["total_store"])
        if do_patch_analysis:
            plot_patches(conds["number_of_patches"])
        if do_charge_analysis:
            plot_charges(conds["number_of_patches"])
        if do_energy_analysis or do_patch_analysis or do_charge_analysis:
            show_plots()

"""

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    filename='test.json'
    params = sim_params(save_filename=filename)
    Engine(params).run()
    Animator(params).animate()
    
