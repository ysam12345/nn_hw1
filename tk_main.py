#!/usr/bin/env python
#coding:utf-8
import numpy as np
from tkinter import *
import tkinter as tk
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter.filedialog import askopenfilename
import pandas as pd
from neural import Neural


class Application(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self, master)
        self.root = master
        self.grid()
        self.create_widgets()
    
    def create_widgets(self):
        self.label = tk.Label(self)
        self.label["text"] = "載入資料集並進行訓練及測試"
        self.label.grid(row=0, column=0, sticky=tk.N+tk.W)

        # 選取資料集檔案
        self.load_data_button = tk.Button(self)
        self.load_data_button["text"] = "選取檔案"
        self.load_data_button.grid(row=0, column=1, sticky=tk.N+tk.W)
        self.load_data_button["command"] = self.select_file_and_run

        # 設定訓練循環次數
        self.epoch_label = tk.Label(self)
        self.epoch_label["text"] = "訓練循環次數"
        self.epoch_label.grid(row=1, column=0, sticky=tk.N+tk.W)

        self.epoch_spinbox = tk.Spinbox(self, from_=1, to=1000)
        self.epoch_spinbox.grid(row=1, column=1, sticky=tk.N+tk.W)
        
        # 設定提前結束條件
        self.early_stop_label = tk.Label(self)
        self.early_stop_label["text"] = "提前結束訓練正確率(%)"
        self.early_stop_label.grid(row=2, column=0, sticky=tk.N+tk.W)

        self.early_stop_spinbox = tk.Spinbox(self, from_=1, to=100)
        self.early_stop_spinbox.grid(row=2, column=1, sticky=tk.N+tk.W)


        # 設定訓練圖表
        self.training_acc_figure = Figure(figsize=(3,3), dpi=100)
        self.training_acc_canvas = FigureCanvasTkAgg(self.training_acc_figure, self)
        self.training_acc_canvas.draw()
        self.training_acc_canvas.get_tk_widget().grid(row=3, column=0, columnspan=3)

        self.training_data_figure = Figure(figsize=(3,3), dpi=100)
        self.training_data_canvas = FigureCanvasTkAgg(self.training_data_figure, self)
        self.training_data_canvas.draw()
        self.training_data_canvas.get_tk_widget().grid(row=3, column=4, columnspan=3)

        # 相關結果文字
        self.training_acc_label = tk.Label(self)
        self.training_acc_label["text"] = "訓練辨識率(%)"
        self.training_acc_label.grid(row=4, column=0, sticky=tk.N+tk.W)

        self.training_acc_text_label = tk.Label(self)
        self.training_acc_text_label["text"] = ""
        self.training_acc_text_label.grid(row=4, column=1, sticky=tk.N+tk.W)

        self.testing_acc_label = tk.Label(self)
        self.testing_acc_label["text"] = "測試辨識率(%)"
        self.testing_acc_label.grid(row=5, column=0, sticky=tk.N+tk.W)

        self.testing_acc_text_label = tk.Label(self)
        self.testing_acc_text_label["text"] = ""
        self.testing_acc_text_label.grid(row=5, column=1, sticky=tk.N+tk.W)

        self.weight_label = tk.Label(self)
        self.weight_label["text"] = "當前鍵結值"
        self.weight_label.grid(row=6, column=0, sticky=tk.N+tk.W)

        self.weight_text = tk.Text(self)
        self.weight_text["height"] = 10
        self.weight_text["width"] = 40
        self.weight_text.grid(row=6, column=1, sticky=tk.N+tk.W)



    def draw_training_acc_figure(self, train_result_list):
        #清空影像
        self.training_acc_figure.clf()
        self.training_acc_figure.a = self.training_acc_figure.add_subplot(111)
        acc_list = [i['acc'] for i in train_result_list]
    
        #繪製正確率折線圖
        self.training_acc_figure.a.plot(acc_list)
        self.training_acc_figure.a.set_title('Traing: Acc/Epoch')
        self.training_acc_canvas.draw()


    def draw_training_data_figure(self, dataset):
        #清空影像
        self.training_data_figure.clf()
        self.training_data_figure.a = self.training_data_figure.add_subplot(111)
        X = dataset[0].values.reshape(-1,).tolist()
        y = dataset[1].values.reshape(-1,).tolist()
        print(dataset)
        print(X)
        print(y)
        #繪製正確率折線圖
        self.training_data_figure.a.plot(X, y, 'ro')
        self.training_data_figure.a.set_title('Traing Data')
        self.training_data_canvas.draw()


        
    def select_file_and_run(self):
        filename = askopenfilename()
        df = pd.read_table(filename, sep=" ", header=None)

        self.draw_training_data_figure(df)

        # split traning data and testing data
        train_df=df.sample(frac=0.666666)
        test_df=df.drop(train_df.index)

        train_dfs = np.split(train_df, [len(train_df.columns)-1], axis=1)
        train_X_df = train_dfs[0]
        train_y_df = train_dfs[1]
        train_X = train_X_df.values.tolist()
        train_y = train_y_df.values.reshape(-1,).tolist()

        test_dfs = np.split(test_df, [len(test_df.columns)-1], axis=1)
        test_X_df = test_dfs[0]
        test_y_df = test_dfs[1]
        test_X = test_X_df.values.tolist()
        test_y = test_y_df.values.reshape(-1,).tolist()

        learning_rate = 0.8

        # run training and show result
        n = Neural(train_X, train_y, learning_rate)
        train_result_list = []
        print("### training start ###")
        for i in range(int(self.epoch_spinbox.get())):
            train_result = n.train()
            train_result_list.append(train_result)
            if train_result["acc"] > int(self.early_stop_spinbox.get()):
                break
        print("### training end ###")
        self.draw_training_acc_figure(train_result_list)
        self.training_acc_text_label["text"] = train_result_list[len(train_result_list)-1]["acc"]
        self.weight_text.delete(1.0, END) 
        self.weight_text.insert(1.0, train_result_list[len(train_result_list)-1]["weight"]) 
        # run testing and show result
        print("### predict start ###")
        test_result = n.test(test_X, test_y)
        print("### predict end ###")

        self.testing_acc_text_label["text"] = test_result["acc"]


root = tk.Tk()
app = Application(root)
root.mainloop()