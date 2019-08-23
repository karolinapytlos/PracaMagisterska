import statistics
from datasets.utils import Dataset
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

class Statistics:
    def __init__(self):
        self.__datasets = []
        self.__data_frame_source = []
        self.__columns = columns=['Dataset', 'Number of rows', 'The longest sequence', 'Average sequence', 'Proportion of classes'
                                  ,'First class', 'First class proportion', 'Second class', 'Second class proportion']
        self.__bar_colors = ["orange", "green", "lightblue", "pink", "crimson", "yellow",
                             "lightgreen", "lightslategray", "teal", "gold", "purple", "aquamarine"]
        self.__chart_layout = go.Layout(
            autosize=False,
            width=1000,
            height=600,
            margin=go.layout.Margin(
                l=70,
                r=50,
                b=50,
                t=70,
                pad=15
            )
        )
        self.__table_layout = go.Layout(
            autosize=False,
            width=1000,
            height=600,
            margin=go.layout.Margin(
                l=50,
                r=50,
                b=50,
                t=70,
                pad=15
            )
        )

    def add_dataset (self, dataset):
        if type(dataset) == Dataset:
            self.__datasets.append(dataset)
            self.__create_data_frame_source(dataset)

    def apply_number_of_rows (self):
        for dataset in self.__datasets:
            self.__apply(dataset, "Number of rows", Statistics.number_of_records)

    def apply_longest_sequence (self):
        for dataset in self.__datasets:
            self.__apply(dataset, "The longest sequence", Statistics.longest_sequence)

    def apply_average_sequence (self):
        for dataset in self.__datasets:
            self.__apply(dataset, "Average sequence", Statistics.average_sequence)

    def apply_proportion_of_classes (self):
        for dataset in self.__datasets:
            self.__apply(dataset, "Proportion of classes", Statistics.proportion_of_classes)

    def display_number_of_rows (self):
        if len(self.__data_frame_source) > 0:
            df = pd.DataFrame(self.__data_frame_source, columns=self.__columns)
            self.__display_bar_chart(df["Dataset"], df["Number of rows"], "Number of rows")
            self.__display_table(df, ["Dataset", "Number of rows"], "Number of rows")

    def display_longest_sequence (self):
        if len(self.__data_frame_source) > 0:
            df = pd.DataFrame(self.__data_frame_source, columns=self.__columns)
            self.__display_bar_chart(df["Dataset"], df["The longest sequence"], "The longest sequence")
            self.__display_table(df, ["Dataset", "The longest sequence"], "The longest sequence")

    def display_average_sequence (self):
        if len(self.__data_frame_source) > 0:
            df = pd.DataFrame(self.__data_frame_source, columns=self.__columns)
            self.__display_bar_chart(df["Dataset"], df["Average sequence"], "Average sequence")
            self.__display_table(df, ["Dataset", "Average sequence"], "Average sequence")

    def display_proportion_of_classes (self):
        if len(self.__data_frame_source) > 0:
            for item in self.__data_frame_source:
                for classes in item:
                    if type(classes) == list:
                        for index, label in enumerate(classes):
                            if index == 0:
                                item[self.__columns.index("First class")] = label[0]
                                item[self.__columns.index("First class proportion")] = label[1]
                            else:
                                item[self.__columns.index("Second class")] = label[0]
                                item[self.__columns.index("Second class proportion")] = label[1]

            df = pd.DataFrame(self.__data_frame_source, columns=self.__columns)
            self.__display_bar_chart_group(df["Dataset"], df["First class proportion"], df["Second class proportion"],
                                           "First class", "Second class",
                                           df["First class"], df["Second class"],
                                           df["First class proportion"], df["Second class proportion"], "Proportion of classes")
            self.__display_table(df, ["Dataset", "First class", "First class proportion",
                                      "Second class", "Second class proportion"], "Proportion of classes")


    def __display_bar_chart (self, x, y, title):
        fig = go.Figure(data=[
            go.Bar(x=x, y=y, text=y, textposition='auto', marker_color=self.__bar_colors)
        ], layout=self.__chart_layout)
        fig.update_layout(title_text=title)
        fig.show()

    def __display_bar_chart_group (self, x, yA, yB, columnA, columnB, hovertextA, hovertextB, textA, textB, title):
        fig = go.Figure(data=[
            go.Bar(name=columnA, x=x, y=yA, hovertext=hovertextA, text=textA, textposition='auto'),
            go.Bar(name=columnB, x=x, y=yB, hovertext=hovertextB, text=textB, textposition='auto')
        ], layout=self.__chart_layout)
        fig.update_layout(title_text=title)
        fig.update_layout(barmode='group')
        fig.show()

    def __display_table (self, data_frame, columns, title):
        fig = go.Figure(data=[go.Table(
            header=dict(values=columns, fill_color='AntiqueWhite', align='left', line_color='Gainsboro'),
            cells=dict(values=[data_frame[column] for column in columns], fill_color='GhostWhite', align='left',
                       line_color='Gainsboro', height=30))
        ], layout=self.__table_layout)
        fig.update_layout(title_text=title)
        fig.show()

    def __create_data_frame_source (self, dataset):
        if len(self.__data_frame_source) > 0:
            if len(list(filter(lambda item: item[0] == dataset.name, self.__data_frame_source))) < 1:
                self.__data_frame_source.append([dataset.name, 0, 0 ,0, 0, "", 0, "", 0])
        else:
            self.__data_frame_source.append([dataset.name, 0, 0 ,0, 0, "", 0, "", 0])

    def __apply (self, dataset, column_name, function):
        ds = list(filter(lambda item: item[0] == dataset.name, self.__data_frame_source))
        if len(ds) < 1:
            return
        ds = ds[0]
        ds[self.__columns.index(column_name)] = function(dataset)
       

    @staticmethod
    def longest_sequence (dataset):
        if type(dataset) == Dataset:
            return max(list(map(len, dataset.X)))
        return 0

    @staticmethod
    def average_sequence (dataset):
        if type(dataset) == Dataset:
            return round(statistics.mean(list(map(len, dataset.X))),2)
        return 0

    @staticmethod
    def number_of_records (dataset):
        if type(dataset) == Dataset:
            return len(dataset.X)
        return 0

    @staticmethod
    def proportion_of_classes (dataset):
        classes = []
        if type(dataset) == Dataset:
            all = len(dataset.y)
            for label in set(dataset.y):
                count = len(list(filter(lambda item: item == label, dataset.y)))
                classes.append((label, str(round((100 * count) / all)) + "%"))
        return classes