"""  
Copyright (c) [2025] [Ye Zhiwei, ]  Dalian University of Technology
All rights reserved.  

This file is part of Deep recongnition of Moleculer fluorescence.  

This code is licensed under the [MIT] 
You may not use this file except in compliance with the License.   

"""  


import tkinter as tk  
from tkinter import filedialog, ttk, messagebox 
import pandas as pd  
import numpy as np
import tifffile as tiff 

class DataLoaderApp:
    def __init__(self):
        # Create GUI  
        self.root = tk.Tk()
        self.data = None # Store data here!
        self.root.title("Reconstruction")  
        
        self.filepath = "d:\\"#defaul address

        self.width = tk.IntVar(value=295)  # default width
        self.height = tk.IntVar(value=512) # default height
        self.pixel = tk.DoubleVar(value=160) #default pixel size.

        self.reconstimg = None # store reconstruction here
        
        self.create_widgets()  
        buttonframe = tk.Frame(self.root)
        buttonframe.pack(pady=5)
        load_btn = tk.Button(buttonframe, text="Load Data", command=self.load_data)  
        load_btn.pack(side=tk.LEFT,padx=20, pady=5) 
        show_btn = tk.Button(buttonframe, text="Show Data", command=self.show_data)
        show_btn.pack(side=tk.RIGHT,padx=20, pady=5)
        reconstruction_btn = tk.Button(buttonframe, text="Reconstruction", command=self.reconstruction)
        reconstruction_btn.pack(padx=20, pady=5)
        save_btn = tk.Button(self.root, text="Save Result",command=self.save)
        save_btn.pack(padx=20,pady=5)

    def load_data(self):  
        filetypes = (  
            ('CSV files', '*.csv'),  
            ('All files', '*.*')  
        )  
        
        filepath = filedialog.askopenfilename(  
            title="Select data file",  
            initialdir="/",  
            filetypes=filetypes  
        )  
        self.filepath = filepath
        
        if filepath:  
            try:  
                if filepath.endswith('.csv'):  
                    self.data = pd.read_csv(filepath)  
                # Add more file type handlers  
                print("Data loaded successfully!")  
            except Exception as e:  
                print(f"Error loading file: {e}")  

    def show_data(self):  
        if self.data is not None:  
            print(self.data.head())  
        else:  
            print("No data loaded!")  

    def create_widgets(self):  
        # 创建输入区域框架  
        input_frame = ttk.LabelFrame(self.root, text="Image Dimensions")  
        input_frame.pack(padx=10, pady=10, fill="x")  
        
        # 宽度输入  
        ttk.Label(input_frame, text="Width:").grid(row=0, column=0, padx=5, pady=5)  
        width_entry = ttk.Entry(  
            input_frame,   
            textvariable=self.width,  
            validate="key",  
            validatecommand=(self.root.register(self.validate_number), '%P')  
        )  
        width_entry.grid(row=0, column=1, padx=5, pady=5)  
        
        # 高度输入  
        ttk.Label(input_frame, text="Height:").grid(row=1, column=0, padx=5, pady=5)  
        height_entry = ttk.Entry(  
            input_frame,  
            textvariable=self.height,  
            validate="key",  
            validatecommand=(self.root.register(self.validate_number), '%P')  
        )  
        height_entry.grid(row=1, column=1, padx=5, pady=5)  

        # pixel size input
        ttk.Label(input_frame, text="pixel size (nm):").grid(row=2, column=0, padx=5, pady=5)
        ADU_entry = ttk.Entry(
            input_frame,  
            textvariable = self.pixel,
            validate="key",
            validatecommand=(self.root.register(self.validate_float),'%P') 
        )
        ADU_entry.grid(row=2, column=1, padx=5, pady=5)

    def validate_number(self, input_text):  
        """验证输入是否为有效数字"""  
        if input_text.strip() == "":  
            return True  
        try:  
            int(input_text)  
            return True  
        except ValueError:  
            messagebox.showerror("Invalid Input", "Please enter a valid integer")  
            return False  
            
    def validate_float(self, input_text):  
        """验证输入是否为有效数字"""  
        if input_text.strip() == "":  
            return True  
        try:  
            float(input_text)  
            return True  
        except ValueError:  
            messagebox.showerror("Invalid Input", "Please enter a valid integer")  
            return False  
            
    def reconstruction(self):
        # reconstructing the images
        numofframes = max(self.data.iloc[:,0])
        #print(numofframes)
        reconstimgs = np.full((self.width.get(),self.height.get(),int(numofframes)),0,dtype=np.int32)
        frames = self.data.iloc[:, 0].to_numpy()  
        x_coords = self.data.iloc[:, 1].to_numpy()  
        y_coords = self.data.iloc[:, 2].to_numpy()  

        # 向量化计算索引（避免逐行计算）  
        x = np.round(x_coords / self.pixel.get()).astype(int)  
        y = np.round(y_coords / self.pixel.get()).astype(int)  
        fr = (np.floor(frames) - 1).astype(int)  

        # 确保索引不越界  
        valid = (x >= 0) & (x < self.width.get()) & (y >= 0) & (y < self.height.get()) & (fr >= 0) & (fr < int(np.max(frames)))  
        x_valid = x[valid]  
        y_valid = y[valid]  
        fr_valid = fr[valid]  
    
        # 一次性计算所有位置的累加（使用向量化操作）  
        reconstimgs = np.zeros((self.width.get(), self.height.get(), int(np.max(frames))), dtype=np.int32)  
        np.add.at(reconstimgs, (x_valid, y_valid, fr_valid), 1)  
    
        self.reconstimg = reconstimgs

    def select_save_location(self):  
        # Open a file dialog to select the save location  
        file_path = filedialog.asksaveasfilename(
            initialdir= self.filepath,
            defaultextension=".tif",   
            filetypes=[("TIFF files", "*.tif;*.tiff"),  
            ("All Files", "*.*")])  

        if file_path:  # Check if a file path was selected  
            return file_path
        else:
            return None


    def save(self):
        savepath = self.select_save_location()
        if savepath:
            #print("Shape of the array:", self.reconstimg.shape)  # Prints the dimensions  
            #print(savepath)
            #print("Shape of the array:",  self.reconstimg.shape)  
            self.reconstimg = self.reconstimg.transpose(1,0,2)
            #print("Shape of the array after transpose:",  self.reconstimg.shape)  
            for index in range(self.reconstimg.shape[2]):
                crsavepath = savepath[:-4] + str(index) + savepath[-4:]
                tiff.imwrite(crsavepath, self.reconstimg[:,:,index])  
            print("Successfully write reconstruction to file!")
        else:
            print("No save path is defined!")


    def run(self):  
        self.root.mainloop()  

if __name__ == "__main__":  
    app = DataLoaderApp()  
    app.run()  