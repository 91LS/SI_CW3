from tkinter import *
from tkinter import ttk
from tkinter import Toplevel
import tools


class PredictionMatrixDialog:
    def __init__(self, parent, trn_system, tst_system):
        top = self.top = Toplevel(parent)
        tree = ttk.Treeview(top, selectmode=NONE)
        scrollbar_y = Scrollbar(top)
        scrollbar_y.pack(side=RIGHT, fill=Y)
        scrollbar_x = Scrollbar(top, orient=HORIZONTAL)
        scrollbar_x.pack(side=BOTTOM, fill=X)

        tree.heading('#0', text='Class symbol')
        trn_classes = tools.get_classes(trn_system)
        columns = trn_classes + ["Count", "Acc.", "Cov."]
        tree["columns"] = columns
        for column in columns:
            tree.column(column, width=50)
            tree.heading(column, text=column)

        scrollbar_y.configure(command=tree.yview)
        tree.configure(yscrollcommand=scrollbar_y.set)
        scrollbar_x.configure(command=tree.xview)
        tree.configure(xscrollcommand=scrollbar_x.set)

        matrix = []
        tst_classes = tools.get_classes(tst_system)
        for tst_class in tst_classes:
            matrix.append(tools.get_values(tst_class, trn_classes, tst_system))
        tools.transform_last_column_to_row(matrix)

        counter = 0
        for tst_class in tst_classes:
            tree.insert("", counter, text=tst_class, values=["{0:0.2f}".format(i) for i in matrix[counter]])
            counter += 1
        tree.insert("", counter, text="True Positive Rate", values=["{0:0.2f}".format(i) for i in matrix[counter]])
        tree.insert("", counter+1, counter+1, text="Global")
        global_data = tools.get_global(tst_system)
        tree.insert(counter+1, counter + 2, text="Acc. = {0:0.2f}".format(global_data[0]))
        tree.insert(counter + 1, counter + 3, text="Cov. = {0:0.2f}".format(global_data[1]))
        tree.pack(fill=BOTH, expand=True)


def center(win):
    """Center dialogs"""
    win.update_idletasks()
    width = win.winfo_width()
    height = win.winfo_height()
    x = (win.winfo_screenwidth() // 2) - (width // 2)
    y = (win.winfo_screenheight() // 2) - (height // 2)
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
