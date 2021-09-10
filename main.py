from conditions import get_conditions
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
# todo 0.5? 1? what is the stepping? how exactly does the verlet work
# todo optimise for speed (side goal)
# try make everything a self. and see if that's faster than inputting to functions (specifically in particle force calc)
# instead of v1 + v2 do np.add(v1, v2) or try the v1.add(v2)? check speed

# this link laughs in my face
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html


def main():
    conds = get_conditions(filename="conds.txt")
    # from my_tools import offset_finder
    # conds['optimal_offset'] = offset_finder(conds['number_of_patches'])
    # print(conds['optimal_offset'])

    # todo better way of choosing what to do please? True False commenting out is strange
    do_physics = False
    # do_physics = True
    if do_physics:
        print("doing physics...")
        Engine(conds).run()
        print("physics is done - the data_dump, charges, and patches files have been written to")
    else:
        print("kept previous physics - the data_dump, charges, and patches files are unchanged")

    do_animate = False
    # do_animate = True
    if do_animate:
        print("animating....")
        Animator(conds).animate()

    # do_analysis = False
    do_analysis = True
    if do_analysis:
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
