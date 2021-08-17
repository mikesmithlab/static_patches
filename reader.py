import numpy as np
import matplotlib.pyplot as plt
import pygame as pg
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from main import rotate
from main import normalise
from main import sphere_maker
from main import find_rotation_matrix
from main import my_cross


def find_truth(o, n):  # old, new positions
    if (n[0] > 0 and o[0] > 0) or (n[0] < 0 and o[0] < 0) or n[1] > 0 and o[1] > 0 or (n[0] < 0 and o[0] < 0):
        return True
    return False
    # todo this logic can be done better?


class Analyser:
    """reads and analyses"""

    def __init__(self):
        from defaults import Defaults
        d = Defaults()
        self.time_list = np.linspace(0, float(d.getter("time_end")), num=int(d.getter("total_store")))

        try:
            self.data_file = open("data_dump", "r")
        except FileNotFoundError:
            print("You deleted the data_dump file or didn't make it with Engine")
            raise FileNotFoundError
        try:
            self.patch_file = open("patches", "r")
        except FileNotFoundError:
            print("You deleted the patches file or didn't make it with ParticlePatches")
            raise FileNotFoundError
        # todo these^ file reads can go in their respective functions, can't they?

    def plot_energy(self):
        energy_list = np.zeros(np.shape(self.time_list))

        i = -3
        for line in self.data_file:
            if i >= 0:  # get out of the way of the first few lines of non-data
                this_line = line.strip()
                field = this_line.split(",")
                energy_list[i] = float(field[12])
            i += 1
        plt.plot(self.time_list, energy_list)

    def plot_patches(self):
        for t in self.time_list:
            if t == 0:
                self.patch_file.readline()  # get out of the way of the first line of non-data
            next_hit_time = float(self.patch_file.readline())
            # try:
            #     next_hit_time = float(self.patch_file.readline())
            # except ValueError:
            #     finished_patches = True  # need this if I use a while loop instead of a for loop

        # fig_e = plt.figure()
        # mngr_e = plt.get_current_fig_manager()
        # mngr_e.window.setGeometry(475, 175, 850, 545)
        # plt.plot(time_list, energy_list, time_list, theta_dot_list)
        # plt.figure()
        # plt.plot(time_list, pos_list[1, :], time_list, pos_list[0, :], time_list, container_pos_list)


class Animator:
    """reads and animates"""

    def __init__(self):
        from defaults import Defaults
        d = Defaults()

        self.container_radius, self.radius = float(d.getter("container_radius")), float(d.getter("radius"))
        self.small_radius = self.radius / 16
        self.time_between_frames = int(d.getter("time_warp")) * float(d.getter("time_step"))
        n = int(d.getter("number_of_patches"))

        self.data_file = open("data_dump", "r")
        self.patch_file = open("patches", "r")

        self.sphere_quadric = gluNewQuadric()
        self.patch_hit_list = np.zeros([n, 1])
        self.next_hit_time = 0
        self.patch_points = sphere_maker(n)
        self.finished_patches = False

    def update_positions(self, f):  # input f is the frame number
        if f == 0:
            self.data_file = open("data_dump", "r")
            for n in range(3):
                self.data_file.readline()  # get out of the way of the first few lines of non-data
        this_line = self.data_file.readline()
        this_line = this_line.strip()
        field = this_line.split(",")
        time_two_dp = "{:.2f}".format(float(field[1]))
        pg.display.set_caption(f"pyopengl shaker, time = {time_two_dp}s")
        return np.array([float(field[2]), float(field[3]), float(field[4])]), np.array(
            [float(field[5]), float(field[6]), float(field[7])]), np.array(
            [float(field[8]), float(field[9]), float(field[10])]), float(field[11]), field[13]
        # pos = [2, 3, 4]
        # particle_x = [5, 6, 7]
        # particle_z = [8, 9, 10]
        # container_height = 11
        # contact = 13

    def update_patch_hit_list(self, f):  # input f is the frame number
        if self.finished_patches:
            return
        if f == 0:
            try:
                self.patch_file = open("patches", "r")
            except FileNotFoundError:
                print("You deleted the patches file or didn't make it with ParticlePatches. Patches won't have colours")
                self.finished_patches = True
            self.patch_file.readline()  # get out of the way of the first line of non-data
            self.next_hit_time = float(self.patch_file.readline())
        if self.next_hit_time <= f * self.time_between_frames:  # <= as there could be two 0 iteration lines @ the start
            hit_line = self.patch_file.readline()
            hit_line = hit_line.strip()
            hit_list = np.asarray(hit_line.split(","))
            # self.patch_hit_list = np.zeros(np.shape(hit_list))  # todo don't need this line? is defined in __init__
            # todo faster way of doing this than the method below!
            for i in range(np.shape(hit_list)[0]):
                self.patch_hit_list[i] = float(hit_list[i])
            try:
                self.next_hit_time = float(self.patch_file.readline())
            except ValueError:
                self.finished_patches = True

    def animate(self, total_frames):
        pg.init()

        display = (int(1280 * 3 / 4), int(1024 * 3 / 4))  # 1280 x 1024
        # display = (int(1920 * 3 / 4), int(1080 * 3 / 4))  # 1920 x 1080
        pg.display.set_mode(display, DOUBLEBUF | OPENGL)
        pg.display.set_caption('pyopengl shaker, time = ')
        glMatrixMode(GL_MODELVIEW)
        glClearColor(0.1, 0.1, 0.1, 0.3)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glEnable(GL_COLOR_MATERIAL)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glShadeModel(GL_SMOOTH)

        camera_radius = 2.25 * self.container_radius
        camera_theta = np.arccos(0 / camera_radius)  # todo why do i have (0 / radius) here???
        camera_phi = np.arctan(1 / 1)
        look_at = [0, 0, 0]
        up_vector = [0, 0, 1]
        # for opengl, x is across the screen, y is up the screen, z is into the screen. For me, z is up not into.
        camera_pos = camera_radius * np.array([
            np.cos(camera_phi) * np.sin(camera_theta),
            np.sin(camera_phi) * np.sin(camera_theta),
            np.cos(camera_theta)])
        perspective = 60, (display[0] / display[1]), 0.001 * self.container_radius, 10 * self.container_radius

        left = False
        right = False
        up = False
        down = False
        # pause = False
        for f in range(total_frames):
            # while pause:
            #     for event in pg.event.get():
            #         if event.type == pg.QUIT:
            #             pg.quit()
            #             quit()
            #         if event.type == pg.MOUSEBUTTONUP:
            #             if event.button == 1:
            #                 pause = False
            #     pg.time.wait(1)  # todo doesn't detect if another button is pressed/unpressed while paused!
            # todo put "for event in pg.event.get()" in a function then call it during the pause & after "elapsed time"
            elapsed_time = pg.time.get_ticks()
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    pg.quit()
                    quit()

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
                    # if event.button == 1:  # todo pause?
                    #     pause = True
                    # glScaled(0.95, 0.95, 0.95)
                    camera_pos = camera_radius * normalise(camera_pos)

            # camera rotation rate modifiers shift and ctrl
            # rotate_amount_per_frame = (2 / 1000) * 2 * np.pi  # 60Hz
            rotate_amount_per_frame = (1 / 1000) * 2 * np.pi  # 144Hz
            if pg.key.get_mods() & pg.KMOD_SHIFT:
                rotate_amount_per_frame *= 2
            elif pg.key.get_mods() & pg.KMOD_CTRL:
                rotate_amount_per_frame *= 1 / 8
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

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)  # clear the screen
            glLoadIdentity()  # is_new_collision matrix
            gluPerspective(*perspective)
            # fov in y, aspect ratio, distance to clipping plane close, distance to clipping plane far
            gluLookAt(*camera_pos, *look_at, *up_vector)

            # draw all objects
            pos, particle_x, particle_z, container_height, contact = self.update_positions(f)
            # gluLookAt(*camera_pos, *pos, *up_vector)  # makes u feel sick
            # container back
            self.draw_container(container_height, contact, "back")
            # particle
            self.draw_particle(pos)
            # lumps and/or patches
            # lumps = True
            lumps = False
            patches = True
            # patches = False
            if lumps:
                lump_offset = particle_x.dot(self.radius)
                self.draw_particle_lump(pos + lump_offset, 1, [0.9, 0.1, 0.1, 0.8])
                self.draw_particle_lump(pos - lump_offset, 2, [0.9, 0.1, 0.1, 0.8])
                lump_offset = particle_z.dot(self.radius)
                self.draw_particle_lump(pos + lump_offset, 1, [0.9, 0.1, 0.9, 0.8])
                self.draw_particle_lump(pos - lump_offset, 2, [0.9, 0.1, 0.9, 0.8])
                lump_offset = normalise(my_cross(particle_x, particle_z)).dot(self.radius)
                self.draw_particle_lump(pos + lump_offset, 1, [0.9, 0.6, 0.1, 0.8])
                self.draw_particle_lump(pos - lump_offset, 2, [0.9, 0.6, 0.1, 0.8])
            if patches:
                self.update_patch_hit_list(f)
                j = 0
                transformation_matrix = find_rotation_matrix(particle_x, particle_z).T.dot(self.radius)
                mat_hit_list = np.amax(self.patch_hit_list)
                for patch in self.patch_points:
                    if mat_hit_list == self.patch_hit_list[j]:
                        rgba = [0, 1, 1, 0.8]
                    elif self.patch_hit_list[j] == 0:
                        rgba = [0.1, 0.1, 0.1, 0.8]
                    else:
                        rgba = [1, 0, self.patch_hit_list[j] / max(1, mat_hit_list), 0.8]
                    self.draw_part_patch(pos + transformation_matrix.dot(patch), rgba)
                    j += 1
            # container front
            self.draw_container(container_height, contact, "front")

            pg.display.flip()  # updates the screen with the new frame
            # pg.time.wait(int((1000 / 60) - (pg.time.get_ticks() - elapsed_time)))  # 60Hz
            pg.time.wait(int((1000 / 144) - (pg.time.get_ticks() - elapsed_time)))  # 144Hz

    def draw_particle_lump(self, pos, one_or_two, rgba):
        glPushMatrix()  # saves current matrix

        glTranslatef(*pos)
        glColor4f(*rgba)
        gluSphere(self.sphere_quadric, (one_or_two / 8) * self.radius, 16, 8)

        glPopMatrix()  # restores current matrix

    def draw_part_patch(self, pos, rgba):
        glPushMatrix()  # saves current matrix

        glTranslatef(*pos)
        glColor4f(*rgba)
        gluSphere(self.sphere_quadric, self.small_radius, 8, 4)

        glPopMatrix()  # restores current matrix

    def draw_particle(self, pos):
        glPushMatrix()  # saves current matrix

        glTranslatef(*pos)
        glColor4f(0.1, 0.9, 0.1, 1)
        gluSphere(self.sphere_quadric, self.radius, 32, 16)  # todo pick appropriate slice and stack numbers, 32 & 16?

        glPopMatrix()  # restores current matrix

    def draw_container(self, height, contact, front_or_back):
        glPushMatrix()  # saves current matrix

        # colour the container depending on front or back being rendered, and whether or not there is contact
        if front_or_back == "front":
            glCullFace(GL_BACK)
            if contact == "True":
                rgba = [0.2, 0.4, 0.9, 0.75]
            else:
                rgba = [0.2, 0.2, 0.9, 0.75]
        elif front_or_back == "back":
            glCullFace(GL_FRONT)
            if contact == "True":
                rgba = [0.1, 0.3, 0.4, 1]
            else:
                rgba = [0.1, 0.1, 0.4, 1]
        else:
            raise ValueError("'front' or 'back' please and thanks")

        glColor4f(*rgba)
        glTranslatef(0, 0, height)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  # make this container a mesh
        gluSphere(self.sphere_quadric, self.container_radius, 32, 16)  # todo slice and stack, 32 & 16?
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)  # from meshes back to normal surfaces

        glPopMatrix()  # restores current matrix
