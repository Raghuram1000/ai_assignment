import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import TKinterModernThemes as TKMT
from PIL import Image, ImageTk
from imagedeblur import train_nnet_fun
from search_algorithms.genetic_algorithm import run_genetic
from search_algorithms.hill_climbing import run_hill_climb
from search_algorithms.swarm_optimization import run_particle_swarm
from search_algorithms.differential_evolution import run_Differential


def open_image():
    filename = filedialog.askopenfilename()
    if filename:
        image = Image.open(filename)
        image = image.resize((int(image.size[0] / 2), int(image.size[1] / 2)))
        photo = ImageTk.PhotoImage(image.convert('L').resize((256, 256)))
        app = TKMT.ThemedTKinterFrame("Image Viewer", "sun-valley", "dark")
        app.Label("Deblurring Image")
        label = tk.Label(app.root, image=photo)
        label.image = photo
        label.grid(row=1, column=0)

        app.root.mainloop()


def show_graphs():
    app = TKMT.ThemedTKinterFrame("Graphs", "sun-valley", "dark")
    # open the graphs folder and display the graphs
    # app.Label("Graphs")

    #     app.Label("Graphs")
    # # two frames, one for laplace and one for ssim
    #     app.laplace_frame = app.addLabelFrame("Laplace", row=0, col=0)
    #
    #
    #     # load the 5 laplace images and display them
    #     laplace_img1 = Image.open("laplace_graph/0_laplace.jpg")
    #     laplace_img1 = laplace_img1.resize((int(laplace_img1.size[0] / 2), int(laplace_img1.size[1] / 2)))
    #     laplace_img1 = ImageTk.PhotoImage(laplace_img1)
    #
    #     laplace_img2 = Image.open("laplace_graph/1_laplace.jpg")
    #     laplace_img2 = laplace_img2.resize((int(laplace_img2.size[0] / 2), int(laplace_img2.size[1] / 2)))
    #     laplace_img2 = ImageTk.PhotoImage(laplace_img2)
    #
    #     laplace_img3 = Image.open("laplace_graph/2_laplace.jpg")
    #     laplace_img3 = laplace_img3.resize((int(laplace_img3.size[0] / 2), int(laplace_img3.size[1] / 2)))
    #     laplace_img3 = ImageTk.PhotoImage(laplace_img3)
    #
    #     laplace_img4 = Image.open("laplace_graph/3_laplace.jpg")
    #     laplace_img4 = laplace_img4.resize((int(laplace_img4.size[0] / 2), int(laplace_img4.size[1] / 2)))
    #     laplace_img4 = ImageTk.PhotoImage(laplace_img4)
    #
    #     laplace_img5 = Image.open("laplace_graph/4_laplace.jpg")
    #     laplace_img5 = laplace_img5.resize((int(laplace_img5.size[0] / 2), int(laplace_img5.size[1] / 2)))
    #     laplace_img5 = ImageTk.PhotoImage(laplace_img5)
    #
    #
    #     laplace_img6 = Image.open("laplace_graph/5_laplace.jpg")
    #     laplace_img6 = laplace_img6.resize((int(laplace_img6.size[0] / 2), int(laplace_img6.size[1] / 2)))
    #     laplace_img6 = ImageTk.PhotoImage(laplace_img6)
    #
    #     label1 = tk.Label(app.root, image=laplace_img1)
    #     label1.image = laplace_img1
    #     label1.grid(row=1, column=0)
    #
    #     label2 = tk.Label(app.root, image=laplace_img2, width=100, height=100)
    #     label2.image = laplace_img2
    #     label2.grid(row=1, column=1)
    #
    #     label3 = tk.Label(app.root, image=laplace_img3, width=100, height=100)
    #     label3.image = laplace_img3
    #     label3.grid(row=2, column=0)
    #
    #     label4 = tk.Label(app.root, image=laplace_img4, width=100, height=100)
    #     label4.image = laplace_img4
    #     label4.grid(row=2, column=1)
    #
    #     label5 = tk.Label(app.root, image=laplace_img5, width=100, height=100)
    #     label5.image = laplace_img5
    #     label5.grid(row=3, column=1)
    #
    #     label6 = tk.Label(app.root, image=laplace_img6, width=100, height=100)
    #     label6.image = laplace_img6
    #     label6.grid(row=4, column=0)
    #

    app.root.mainloop()


def buttonCMD():
    print("Button clicked!")


def genetic_algo():
    return 1


def run_search_algorithm():
    app = TKMT.ThemedTKinterFrame("Search Algorithm", "sun-valley", "dark")
    app.image_frame = app.addLabelFrame("Image and Ground Truth", row=0, col=0, colspan=5)
    # add a text box to set image file path and ground truth file path
    image_file_path = tk.Variable()
    image_file_path.set("datasets/levin/im1_kernel1_img.png")
    ground_truth_file_path = tk.Variable()
    ground_truth_file_path.set("datasets/levin/gt/im1.png")
    app.image_frame.Entry(textvariable=image_file_path)
    app.image_frame.Entry(textvariable=ground_truth_file_path)

    app.genetic_frame = app.addLabelFrame("Genetic Algorithm", row=1, col=0)
    app.hill_climb_frame = app.addLabelFrame("Hill Climb", row=1, col=1)
    app.swarm_opt_frame = app.addLabelFrame("Swarm Optimization", row=1, col=2)
    # differential evolution
    app.differential_frame = app.addLabelFrame("Differential Evolution", row=1, col=3)

    # add a text box to set population size, generations, mutation rate, and targeted variance
    pop_size = tk.Variable()
    generations = tk.Variable()
    mutation_rate = tk.Variable()
    targeted_variance_genetic = tk.Variable()
    app.genetic_frame.Label("Population size")
    app.genetic_frame.Entry(textvariable=pop_size)
    app.genetic_frame.Label("Generations")
    app.genetic_frame.Entry(textvariable=generations)
    app.genetic_frame.Label("Mutation rate")
    app.genetic_frame.Entry(textvariable=mutation_rate)
    app.genetic_frame.Label("Targeted variance")
    app.genetic_frame.Entry(textvariable=targeted_variance_genetic)
    genetic_vars = [0, 0, 0, 0]

    def get_genetic_vars():
        genetic_vars[0] = int(pop_size.get())
        genetic_vars[1] = int(generations.get())
        genetic_vars[2] = int(mutation_rate.get())
        genetic_vars[3] = int(targeted_variance_genetic.get())
        image_file = image_file_path.get()
        ground_truth_file = ground_truth_file_path.get()

        print(genetic_vars)
        run_genetic(image_file, ground_truth_file, genetic_vars[3], genetic_vars[0], genetic_vars[1],
                    genetic_vars[2])

    app.genetic_frame.Button("Deblur using genetic algorithm", get_genetic_vars)
    # frame for hill climb and swarm optimization
    # add a text box to set n, step_size, targeted variance, and max iterations

    step_size = tk.Variable()
    targeted_variance_hill_climb = tk.Variable()
    max_iterations = tk.Variable()

    app.hill_climb_frame.Label("Step size")
    app.hill_climb_frame.Entry(textvariable=step_size)
    app.hill_climb_frame.Label("Targeted variance")
    app.hill_climb_frame.Entry(textvariable=targeted_variance_hill_climb)
    app.hill_climb_frame.Label("Max iterations")
    app.hill_climb_frame.Entry(textvariable=max_iterations)
    hill_climb_vars = [0, 0, 0, 0]

    def get_hill_climb_vars():
        hill_climb_vars[0] = 0
        hill_climb_vars[1] = float(step_size.get())
        hill_climb_vars[2] = int(targeted_variance_hill_climb.get())
        hill_climb_vars[3] = int(max_iterations.get())
        image_file = image_file_path.get()
        ground_truth_file = ground_truth_file_path.get()

        print(hill_climb_vars)
        run_hill_climb(blur_img=image_file, clear_img=ground_truth_file, step_size=hill_climb_vars[1],
                       max_iterations=hill_climb_vars[3], targetedVariance_inp=hill_climb_vars[2])

    app.hill_climb_frame.Button("Deblur using hill climb", get_hill_climb_vars)

    kernel_size = tk.Variable()
    targeted_variance_swarm = tk.Variable()
    app.swarm_opt_frame.Label("Kernel size")
    app.swarm_opt_frame.Entry(textvariable=kernel_size)
    app.swarm_opt_frame.Label("Targeted variance")
    app.swarm_opt_frame.Entry(textvariable=targeted_variance_swarm)
    swarm_vars = [0, 0]

    def get_swarm_vars():
        swarm_vars[0] = int(kernel_size.get())
        swarm_vars[1] = int(targeted_variance_swarm.get())
        image_file = image_file_path.get()
        ground_truth_file = ground_truth_file_path.get()

        print(swarm_vars)
        run_particle_swarm(image_file, ground_truth_file, swarm_vars[0], swarm_vars[1])

    app.swarm_opt_frame.Button("Deblur using swarm optimization", get_swarm_vars)

    # differential evolution
    # add a text box to set targeted variance
    targeted_variance_diff = tk.Variable()
    app.differential_frame.Label("Targeted variance")
    app.differential_frame.Entry(textvariable=targeted_variance_diff)
    diff_vars = [0]

    def get_diff_vars():
        diff_vars[0] = int(targeted_variance_diff.get())
        image_file = image_file_path.get()
        ground_truth_file = ground_truth_file_path.get()

        print(diff_vars)
        run_Differential(image_file, ground_truth_file, diff_vars[0])

    app.differential_frame.Button("Deblur using differential evolution", get_diff_vars)

    app.root.mainloop()


def buttonCMD2(func):
    def wrapper():
        func()

    return wrapper


options = [5000, 256, 21, "datasets/levin/", "results/levin/", 100]


def train_nnet():
    app = TKMT.ThemedTKinterFrame("Training Neural Network", "sun-valley", "dark")
    app.Label("Training Neural Network")
    app.num_iter_frame = app.addLabelFrame("Number of iterations")
    default_num_iter = tk.Variable(app.root, value=options[0])
    app.num_iter = app.num_iter_frame.Entry(textvariable=default_num_iter)
    app.img_size_frame = app.addLabelFrame("Image size ( width x height )")
    default_img_size = tk.Variable(app.root, value=options[1])
    app.img_size = app.img_size_frame.Entry(textvariable=default_img_size)
    app.kernel_size_frame = app.addLabelFrame("Kernel size ( width x height )")
    default_kernel_size = tk.Variable(app.root, value=options[2])
    app.kernel_size = app.kernel_size_frame.Entry(textvariable=default_kernel_size)
    app.data_path_frame = app.addLabelFrame("Data path")
    app.data_path_frame.Label("Data path")
    default_data_path = tk.Variable(app.root, value=options[3])
    app.data_path = app.data_path_frame.Entry(textvariable=default_data_path)
    app.data_path_frame.Label("Output path")
    default_output_path = tk.Variable(app.root, value=options[4])
    app.output_path = app.data_path_frame.Entry(textvariable=default_output_path)
    app.batch_size_frame = app.addLabelFrame("Batch size")
    default_batch_size = tk.Variable(app.root, value=options[5])
    app.batch_size = app.batch_size_frame.Entry(textvariable=default_batch_size)
    app.run_frame = app.addLabelFrame("Run")

    def run_nnet():
        options[0] = int(default_num_iter.get())
        options[1] = int(default_img_size.get())
        options[2] = int(default_kernel_size.get())
        options[3] = str(default_data_path.get())
        options[4] = str(default_output_path.get())
        options[5] = int(default_batch_size.get())

        train_nnet_fun(options)

    app.run_frame.Button("Run", run_nnet)
    app.root.mainloop()


def run_app():
    train_nnet()


class App(TKMT.ThemedTKinterFrame):
    def __init__(self, theme, mode, usecommandlineargs=True, usethemeconfigfile=True):
        super().__init__(str("Blur removal"), theme, mode, usecommandlineargs, usethemeconfigfile)
        self.button_frame = self.addLabelFrame(str("Main Functions"))  # placed at row 1, col 0
        self.button_frame.Button(str("Train Neural Network"), buttonCMD2(lambda: run_app()))
        self.button_frame.Button(str("Deblur an image (Using search algorithm) "), open_image)
        self.button_frame_2 = self.addLabelFrame(str("Analysis"))  # placed at row 1, col 0
        self.button_frame_2.Button(str("Run Search algorithms"), buttonCMD2(lambda: run_search_algorithm()))
        # self.button_frame_2.Button(str("Compare search algorithms"), buttonCMD2(lambda: show_graphs()))
        self.debugPrint()
        self.run()


if __name__ == "__main__":
    App(str("park"), str("dark"))
