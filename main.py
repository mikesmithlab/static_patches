from conditions import get_conditions
from objects import Engine
from reader import Animator, plot_energy, plot_patches

# todo
# air resistance? linear and/or rotational ---- do calculations
# tiny tiny bit of randomness in the collision forces (more realistic)?
# measure all physical parameters:
# container_amplitude
# container_radius
# density
# coefficient_of_restitution
# gamma_t (viscous damping coefficient)
# mu (coefficient of friction)
# todo 0.5? 1? what is the stepping? how exactly does the verlet work
# todo optimise for speed (side goal)
# try make everything a self. and see if that's faster than inputting to functions (specifically in particle force calc)
# instead of v1 + v2 do np.add(v1, v2) or try the v1.add(v2)? check speed


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    conds = get_conditions(filename="conds.txt")  # todo a "try" here? have an "except FileNotFound try data_dump"
    # from my_tools import offset_finder
    # conds['optimal_offset'] = offset_finder(conds['number_of_patches'])
    # print(conds['optimal_offset'])

    # todo better way of choosing what to do please? True False commenting out is strange
    do_physics = False
    # do_physics = True
    if do_physics:
        print("doing physics...")
        Engine(conds).run()
        print("physics is done - data_dump has been written to")
    else:
        print("kept previous physics - data_dump is unchanged")

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
        # do_patch_analysis = False
        do_patch_analysis = True
        if do_energy_analysis:
            plot_energy(do_patch_analysis, conds["time_end"], conds["total_store"])
        if do_patch_analysis:
            plot_patches(conds["number_of_patches"])
