import numpy as np
import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

from my_tools import rotate, normalise, find_rotation_matrix, my_cross, sphere_points_maker


def find_truth(o, n):  # logic for camera with inputs as old and new positions  # todo can be done a lot better
    if (n[0] > 0 and o[0] > 0) or (n[0] < 0 and o[0] < 0) or n[1] > 0 and o[1] > 0 or (n[0] < 0 and o[0] < 0):
        return True
    return False


def find_rgba(hits, max_hits):  # returns an rgb colour and transparency for patches based on its number of hits
    if hits == 0:
        return [0.1, 0.1, 0.1, 0.3]
    elif max_hits == hits:
        return [0, 1, 1, 1]
    else:
        return [1, 0, hits / max(1, max_hits), 0.8]  # avoid a divide by 0 using max


class Animator:
    """
    Reads from the data_dump file produced by Engine and animates the system
    """

    def __init__(self, conds):
        self.container_radius = conds["container_radius"]
        self.radius = conds["radius"]
        self.small_radius = self.radius / 16
        n = conds["number_of_patches"]
        self.total_store = conds["total_store"]
        self.time_between_frames = conds["store_interval"] * conds["time_step"]
        self.refresh_rate = conds["refresh_rate"]

        try:
            open("data_dump", "r")
        except FileNotFoundError:
            raise FileNotFoundError("You deleted the data_dump file or didn't make it with Engine. Animation stopped.")
        try:
            open("patches", "r")
            self.finished_patches = False
        except FileNotFoundError:
            print("You deleted the patches file or didn't make it with PatchTracker. Patches won't have colours")
            self.finished_patches = True
        try:
            open("charges", "r")
            self.finished_charges = False
        except FileNotFoundError:
            print("You deleted the charges file or didn't make it with PatchTracker. Patches won't have colours")
            self.finished_charges = True

        self.sphere_quad = gluNewQuadric()

        self.patch_points = sphere_points_maker(n, conds["optimal_offset"])

        self.patch_hit_list = np.zeros([n, 2])
        self.patch_charge_list = np.zeros([n, 2])
        self.data_file_line = 3  # non-zero to pass over the first few lines of non-data
        self.patch_file_line = 1
        self.charge_file_line = 1

    def update_positions(self):  # returns animation data from line f of data_dump
        with open("data_dump", "r") as data_file:
            for _ in range(self.data_file_line):  # skip past any already read line
                data_file.readline()
            field = data_file.readline().strip().split(",")
        self.data_file_line += 1
        time_two_dp = "{:.2f}".format(float(field[1]))
        pg.display.set_caption(f"pyopengl shaker, time = {time_two_dp}s")
        return np.array([float(field[2]), float(field[3]), float(field[4])]), np.array(
            [float(field[5]), float(field[6]), float(field[7])]), np.array(
            [float(field[8]), float(field[9]), float(field[10])]), float(field[11]), field[13]
        # pos = [2, 3, 4]
        # particle_x_axis = [5, 6, 7]
        # particle_z_axis = [8, 9, 10]
        # container_pos = 11
        # contact = 13

    def update_patch_hit_list(self, f):  # input f is the frame number
        if self.finished_patches:
            return
        with open("patches", "r") as patch_file:
            for _ in range(self.patch_file_line):  # skip past any already read line
                patch_file.readline()
            if float(patch_file.readline()) <= f * self.time_between_frames:  # if next time <= animation time
                field = patch_file.readline().strip().split(",")
                for i in [0, 1]:  # for particle and container
                    self.patch_hit_list[int(field[i]), i] += 1  # +1 to number of collisions for this patch
                try:
                    float(patch_file.readline())
                except ValueError:  # todo better catch than "try except"? Like "if empty"?
                    self.finished_patches = True
                self.patch_file_line += 2

    def update_charges_list(self, f):
        if self.finished_charges:
            return
        with open("charges", "r") as charge_file:
            for _ in range(self.charge_file_line):  # skip past any already read line
                charge_file.readline()
            if float(charge_file.readline()) <= f * self.time_between_frames:  # if next time <= animation time
                for i in [0, 1]:  # for particle and container
                    self.patch_charge_list[:, i] = np.array(charge_file.readline().strip().split(","))  # set charges
                try:
                    float(charge_file.readline())
                except ValueError:  # todo better catch than "try except"? Like "if empty"?
                    self.finished_charges = True
                self.charge_file_line += 3

    def animate(self):
        # ----------------
        # setting up the scene
        pg.init()

        display = (int(1280 * 3 / 4), int(1024 * 3 / 4))  # 1280 x 1024
        pg.display.set_mode(display, DOUBLEBUF | OPENGL)
        pg.display.set_caption('pyopengl shaker, time = ')
        glMatrixMode(GL_MODELVIEW)
        glClearColor(0.1, 0.1, 0.1, 0.3)
        # --------
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glEnable(GL_COLOR_MATERIAL)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)

        # ----------------
        # camera initial conditions
        camera_radius = 2.25 * self.container_radius
        camera_theta = np.arccos(0 / camera_radius)  # why do i have (0 / radius) here???
        camera_phi = np.arctan(1 / 1)
        look_at = [0, 0, 0]
        up_vector = [0, 0, 1]
        # for opengl, x is across the screen, y is up the screen, z is into the screen. For me, z is up not into.
        camera_pos = camera_radius * np.array([
            np.cos(camera_phi) * np.sin(camera_theta),
            np.sin(camera_phi) * np.sin(camera_theta),
            np.cos(camera_theta)])
        perspective = 60, (display[0] / display[1]), 0.001 * self.container_radius, 10 * self.container_radius
        # fov in y, aspect ratio, distance to clipping plane close, distance to clipping plane far

        left = False
        right = False
        up = False
        down = False
        # pause = False
        for f in range(self.total_store - 1):  # todo why does there need to be a -1 here?
            # while pause:
            #     for event in pg.event.get():
            #         if event.type == pg.QUIT:
            #             pg.quit()
            #             quit()
            #         if event.type == pg.MOUSEBUTTONUP:
            #             if event.button == 1:
            #                 pause = False
            #     pg.time.wait(1)  # doesn't detect if another button is pressed/unpressed while paused!
            # todo put "for event in pg.event.get()" in a function then call it during the pause & after "elapsed time"
            elapsed_time = pg.time.get_ticks()

            # ----------------
            # do camera controls
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    quit()
                # --------
                # arrow key press: start rotation
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_LEFT or event.key == pg.K_a:
                        left = True
                    if event.key == pg.K_RIGHT or event.key == pg.K_d:
                        right = True
                    if event.key == pg.K_UP or event.key == pg.K_w:
                        up = True
                    if event.key == pg.K_DOWN or event.key == pg.K_s:
                        down = True
                # --------
                # arrow key lift: stop rotation
                if event.type == pg.KEYUP:
                    if event.key == pg.K_LEFT or event.key == pg.K_a:
                        left = False
                    if event.key == pg.K_RIGHT or event.key == pg.K_d:
                        right = False
                    if event.key == pg.K_UP or event.key == pg.K_w:
                        up = False
                    if event.key == pg.K_DOWN or event.key == pg.K_s:
                        down = False
                # --------
                # mouse wheel zoom
                if event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 4:  # wheel rolled up
                        camera_radius = 0.95 * camera_radius
                        # camera_radius -= 0.05 * self.container_radius
                    if event.button == 5:  # wheel rolled
                        camera_radius = 1.05 * camera_radius
                        # camera_radius += 0.05 * self.container_radius
                    if event.button == 2:  # wheel press
                        camera_radius = 2.25 * self.container_radius
                    # if event.button == 1:
                    #     pause = True
                    # glScaled(0.95, 0.95, 0.95)
                    camera_pos = camera_radius * normalise(camera_pos)
            # --------
            # camera rotation rate modifiers shift and ctrl
            rotate_amount_per_frame = self.refresh_rate * 1e-5 * 2 * np.pi
            if pg.key.get_mods() & pg.KMOD_SHIFT:
                rotate_amount_per_frame *= 2
            elif pg.key.get_mods() & pg.KMOD_CTRL:
                rotate_amount_per_frame *= 1 / 8
            # --------
            # check for all arrow key rotation
            if left:
                camera_pos = rotate(np.array([0, 0, -rotate_amount_per_frame]), camera_pos)
            if right:
                camera_pos = rotate(np.array([0, 0, rotate_amount_per_frame]), camera_pos)
            if up:
                new_camera_pos = rotate(
                    rotate_amount_per_frame * normalise(my_cross(camera_pos, np.array([0, 0, 1]))), camera_pos)
                if find_truth(camera_pos, new_camera_pos):
                    camera_pos = new_camera_pos
            if down:
                new_camera_pos = rotate(
                    rotate_amount_per_frame * normalise(my_cross(np.array([0, 0, 1]), camera_pos)), camera_pos)
                if find_truth(camera_pos, new_camera_pos):
                    camera_pos = new_camera_pos

            # ----------------
            # do camera
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # clear the screen
            glLoadIdentity()  # new matrix
            gluPerspective(*perspective)
            gluLookAt(*camera_pos, *look_at, *up_vector)

            # ----------------
            # update positions from file
            pos, particle_x, particle_z, container_height, contact = self.update_positions()
            # gluLookAt(*camera_pos, *pos, *up_vector)  # makes you feel sick, focuses on the particle with the camera

            # ----------------
            # draw all objects
            # --------
            # container back
            self.draw_container(container_height, contact, "back")
            # --------
            # particle
            self.draw_particle(pos)
            # --------
            # charges/patches
            # charges = False
            charges = True
            if charges:
                self.update_charges_list(f)
                self.draw_patch_centre_spheres(pos, np.array([0, 0, container_height]), self.patch_charge_list,
                                               find_rotation_matrix(particle_x, particle_z).T.dot(self.radius))
            else:
                self.update_patch_hit_list(f)
                self.draw_patch_centre_spheres(pos, np.array([0, 0, container_height]), self.patch_hit_list,
                                               find_rotation_matrix(particle_x, particle_z).T.dot(self.radius))
            # --------
            # container front
            self.draw_container(container_height, contact, "front")

            # ----------------
            # update the screen with the new frame then pause for the appropriate time between frames
            pg.display.flip()
            pg.time.wait(int((1000 / self.refresh_rate) - (pg.time.get_ticks() - elapsed_time)))

    def draw_patch_centre_spheres(self, pos, c_pos, hits_or_charges, transformation):
        m0 = np.amax(hits_or_charges[:, 0])
        m1 = np.amax(hits_or_charges[:, 1])
        j = 0
        for patch in self.patch_points:  # draw both particle and container patches for every patch
            self.draw_patch_centre_sphere(pos + transformation.dot(patch),
                                          find_rgba(hits_or_charges[j, 0], m0))
            self.draw_patch_centre_sphere(c_pos + patch.dot(self.container_radius),
                                          find_rgba(hits_or_charges[j, 1], m1))
            j += 1

    def draw_particle_lump(self, pos, one_or_two, rgba):  # draws a lump (designed to be one on the + & - of each axis)
        glPushMatrix()  # saves current matrix

        glTranslatef(*pos)
        glColor4f(*rgba)
        gluSphere(self.sphere_quad, (one_or_two / 8) * self.radius, 16, 8)

        glPopMatrix()  # restores current matrix

    def draw_patch_centre_sphere(self, pos, rgba):  # draws the centre of a patch for container or particle
        glPushMatrix()  # saves current matrix

        glTranslatef(*pos)
        glColor4f(*rgba)
        gluSphere(self.sphere_quad, self.small_radius, 8, 4)

        glPopMatrix()  # restores current matrix

    def draw_particle(self, pos):  # draws the particle itself
        glPushMatrix()  # saves current matrix

        glTranslatef(*pos)
        glColor4f(0.1, 0.9, 0.1, 1)
        gluSphere(self.sphere_quad, self.radius, 32, 16)  # pick appropriate slice and stack numbers, 32 & 16?

        glPopMatrix()  # restores current matrix

    def draw_container(self, height, contact, front_or_back):  # draws the container in mesh instead of solid
        glPushMatrix()  # saves current matrix

        # colour the container depending on front or back being rendered, and whether or not there is contact
        if front_or_back == "front":
            glCullFace(GL_BACK)
            if contact == "True":
                glColor4f(0.2, 0.4, 0.9, 0.75)
            else:
                glColor4f(0.2, 0.2, 0.9, 0.75)
        elif front_or_back == "back":
            glCullFace(GL_FRONT)
            if contact == "True":
                glColor4f(0.1, 0.3, 0.4, 1)
            else:
                glColor4f(0.1, 0.1, 0.4, 1)
        else:
            raise ValueError("'front' or 'back' please and thanks")

        glTranslatef(0, 0, height)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  # make this container a mesh
        gluSphere(self.sphere_quad, self.container_radius, 32, 16)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)  # from meshes back to normal surfaces

        glPopMatrix()  # restores current matrix
