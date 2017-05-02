import tools
from tkinter import _setit
from tkinter import *
import dialogs
import tkinter.filedialog as filedialog


class MainFrame(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)

        self.parent = parent
        self.system_file_path = ''
        self.type_filename = ''
        self.__init_ui()

    def __init_ui(self):
        self.parent.title("Decision System Reader")
        self.pack(fill=BOTH, expand=True)

        trn_system_load_frame = Frame(self)  # 1st frame // load trn
        trn_system_load_frame.pack(fill=X)

        self.load_system_button = Button(trn_system_load_frame, text="Load TRN system",
                                         command=self.__get_trn_system_filename, width=15)
        self.load_system_button.pack(side=LEFT, padx=5, pady=5)

        self.trn_system_text_box = Entry(trn_system_load_frame)
        self.trn_system_text_box.pack(fill=X, padx=5, expand=True)
        self.trn_system_text_box.configure(state=DISABLED)

        tst_system_load_frame = Frame(self)  # 2nd frame // load tst
        tst_system_load_frame.pack(fill=X)

        self.load_system_button = Button(tst_system_load_frame, text="Load TST system",
                                         command=self.__get_tst_system_filename, width=15)
        self.load_system_button.pack(side=LEFT, padx=5, pady=5)

        self.tst_system_text_box = Entry(tst_system_load_frame)
        self.tst_system_text_box.pack(fill=X, padx=5, expand=True)
        self.tst_system_text_box.configure(state=DISABLED)

        options_frame = Frame(self)  # 3rd frame // combobox and GO!
        options_frame.pack(fill=X)

        self.measure_label = Label(options_frame, text="Select measure:")
        self.measure_label.pack(side=LEFT, padx=5, pady=5)

        self.measure = StringVar()
        self.measure.set("Canberra")
        self.measure_options = ["Canberra", "Chebyshev", "Euclidean", "Manhattan", "Pearson"]
        self.measure_menu = OptionMenu(options_frame, self.measure, *self.measure_options)
        self.measure_menu.config(width=12)
        self.measure_menu.pack(side=LEFT, padx=5, pady=5)

        self.k_nn_label = Label(options_frame, text="Select k nearest neighbours:")
        self.k_nn_label.pack(side=LEFT, padx=5, pady=5)

        self.k_nn = IntVar()
        self.k_nn.set('')
        self.k_nn_menu = OptionMenu(options_frame, self.k_nn, '')
        self.k_nn_menu.config(width=5)
        self.k_nn_menu.pack(side=LEFT, padx=5, pady=5)

        self.start_button = Button(options_frame, text="SHOW MATRIX", state=DISABLED, command=self.__start_k_nn)
        self.start_button.pack(padx=5, pady=5, fill=X)

    def __get_trn_system_filename(self):
        self.trn_system_file_path = filedialog.askopenfilename(filetypes=[('Txt files', '*.txt')])
        self.trn_system_text_box.configure(state=NORMAL)
        self.trn_system_text_box.delete(0, "end")
        self.trn_system_text_box.insert(0, self.trn_system_file_path)
        self.trn_system_text_box.configure(state=DISABLED)
        if self.trn_system_file_path != '':
            with open(self.trn_system_file_path) as file:
                self.trn_system = tools.get_system_objects(file)
            self.__refresh_list()
        if self.__are_systems_chosen():
            self.start_button.config(state=NORMAL)

    def __get_tst_system_filename(self):
        self.tst_system_file_path = filedialog.askopenfilename(filetypes=[('Txt files', '*.txt')])
        self.tst_system_text_box.configure(state=NORMAL)
        self.tst_system_text_box.delete(0, "end")
        self.tst_system_text_box.insert(0, self.tst_system_file_path)
        self.tst_system_text_box.configure(state=DISABLED)
        if self.tst_system_file_path != '':
            with open(self.tst_system_file_path) as file:
                self.tst_system = tools.get_system_objects(file)
        if self.__are_systems_chosen():
            self.start_button.config(state=NORMAL)

    def __are_systems_chosen(self):
        if self.tst_system_text_box.get() != '' and self.trn_system_text_box.get() != '':
            return True
        else:
            return False

    def __refresh_list(self):
        k_nn_options = list(range(1, tools.get_maximum_k_size(self.trn_system) + 1))
        self.k_nn_menu["menu"].delete(0, "end")
        for number in k_nn_options:
            self.k_nn_menu["menu"].add_command(label=number, command=_setit(self.k_nn, number))
        self.k_nn.set(k_nn_options[0])

    def __start_k_nn(self):
        tools.classify_objects(self.__get_measure(), self.trn_system, self.tst_system, self.k_nn.get())
        self.show_matrix()

    def show_matrix(self):
        tree = dialogs.PredictionMatrixDialog(self.parent, self.trn_system, self.tst_system)
        dialogs.center(tree.top)

    def __get_measure(self):
        if self.measure.get() == "Canberra":
            return tools.get_canberra
        elif self.measure.get() == "Chebyshev":
            return tools.get_chebyshev
        elif self.measure.get() == "Euclidean":
            return tools.get_euclidean
        elif self.measure.get() == "Manhattan":
            return tools.get_manhattan
        elif self.measure.get() == "Pearson":
            return tools.get_pearson


def main():
    main_frame = Tk()
    ex = MainFrame(main_frame)
    main_frame.geometry("660x115+420+300")
    main_frame.mainloop()


if __name__ == '__main__':
    main()
