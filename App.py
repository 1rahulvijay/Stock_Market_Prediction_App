import tkinter as tk
from tkinter import filedialog, messagebox
import logging
import sys
import threading


class TextWidgetHandler(logging.Handler):
    COLORS = {
        "DEBUG": "#2196F3",
        "INFO": "#4CAF50",
        "WARNING": "#FFC107",
        "ERROR": "#FF5722",
        "CRITICAL": "#F44336",
    }

    def __init__(self, text_widget):
        super().__init__()
        self.text_widget = text_widget

    def emit(self, record):
        msg = self.format(record)
        color = self.COLORS.get(record.levelname, "")
        self.text_widget.configure(state="normal")
        self.text_widget.insert(tk.END, msg + "\n", record.levelname)
        self.text_widget.tag_configure(record.levelname, foreground=color)
        self.text_widget.configure(state="disabled")
        self.text_widget.see(tk.END)


class AnotherClass:
    def __init__(self, logger):
        self.logger = logger

    def do_something(self, file1_path, file2_path):
        try:
            # Perform some operations with the files
            self.logger.info(f"Processing files: {file1_path}, {file2_path}")
            # Perform some operations with the files
            self.logger.info(f"Processing files:")
            # Additional processing...
            self.logger.info("jsk")

            # Example of logging with a specific level
            self.logger.debug("Debug message from AnotherClass")
            # Additional processing...
        except Exception as e:
            self.logger.error(f"Error occurred during processing: {e}")


class App:
    def __init__(self, master):
        self.master = master
        self.setup_ui()
        self.setup_logging()

    def setup_logging(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # Create a custom logging handler to display logs in the text widget
        self.log_handler = TextWidgetHandler(self.log_text)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.log_handler.setFormatter(formatter)
        self.logger.addHandler(self.log_handler)

    def setup_ui(self):
        self.master.title("AML - File Selector and Password Input")
        self.master.geometry("900x650")
        self.master.resizable(False, False)

        bg_color = "#FFFFFF"
        text_color = "#333333"
        button_color = "#001F3F"  # Dark Navy Blue
        button_text_color = "#FFFFFF"
        label_bg_color = "#E0E0E0"
        entry_bg_color = "#F5F5F5"
        password_bg_color = "#EEEEEE"

        self.master.configure(bg=bg_color)

        title_label = tk.Label(
            self.master,
            text="AML - File Selector and Password Input",
            font=("Arial", 24, "bold"),
            bg=bg_color,
            fg=button_color,
        )
        title_label.pack(pady=20)

        file_frame = tk.Frame(self.master, bg=label_bg_color)
        file_frame.pack(pady=10)

        file1_label = tk.Label(
            file_frame,
            text="File 1:",
            font=("Arial", 12),
            bg=label_bg_color,
            fg=text_color,
        )
        file1_label.grid(row=0, column=0, pady=5, padx=10)
        self.file1_entry = tk.Entry(
            file_frame, width=50, font=("Arial", 12), bg=entry_bg_color
        )
        self.file1_entry.insert(tk.END, "Enter file 1 path here")
        self.file1_entry.grid(row=0, column=1, padx=5, pady=5)
        file1_button = tk.Button(
            file_frame,
            text="Browse",
            command=lambda: self.browse_file(self.file1_entry),
            bg=button_color,
            fg=button_text_color,
        )
        file1_button.grid(row=0, column=2, padx=5, pady=5)

        file2_label = tk.Label(
            file_frame,
            text="File 2:",
            font=("Arial", 12),
            bg=label_bg_color,
            fg=text_color,
        )
        file2_label.grid(row=1, column=0, pady=5, padx=10)
        self.file2_entry = tk.Entry(
            file_frame, width=50, font=("Arial", 12), bg=entry_bg_color
        )
        self.file2_entry.insert(tk.END, "Enter file 2 path here")
        self.file2_entry.grid(row=1, column=1, padx=5, pady=5)
        file2_button = tk.Button(
            file_frame,
            text="Browse",
            command=lambda: self.browse_file(self.file2_entry),
            bg=button_color,
            fg=button_text_color,
        )
        file2_button.grid(row=1, column=2, padx=5, pady=5)

        password_label = tk.Label(
            self.master,
            text="Password:",
            font=("Arial", 12),
            bg=label_bg_color,
            fg=text_color,
        )
        password_label.pack(pady=5)

        password_frame = tk.Frame(self.master, bg=password_bg_color)
        password_frame.pack()

        self.password_entry = tk.Entry(
            password_frame, width=50, font=("Arial", 12), bg=entry_bg_color, show="*"
        )
        self.password_entry.pack(side=tk.LEFT, padx=5, pady=5)

        self.show_password_var = tk.BooleanVar()
        self.show_password_var.set(False)
        show_password_checkbox = tk.Checkbutton(
            password_frame,
            text="Show Password",
            variable=self.show_password_var,
            font=("Arial", 10),
            bg=password_bg_color,
            command=self.show_password,
        )
        show_password_checkbox.pack(side=tk.LEFT, padx=5, pady=5)

        submit_button = tk.Button(
            self.master,
            text="Submit",
            command=self.submit,
            font=("Arial", 14),
            bg=button_color,
            fg=button_text_color,
        )

        submit_button.pack(pady=20, fill=tk.X)

        # Create a text widget to display logs
        self.log_text = tk.Text(
            self.master,
            height=10,
            width=80,
            font=("Arial", 12),
            wrap=tk.WORD,
            bg=entry_bg_color,
        )
        self.log_text.pack(pady=10)

        # Additional text columns
        self.text_columns_frame = tk.Frame(self.master)
        self.text_columns_frame.pack(pady=10)

        # Text column 1
        self.text_column1_label = tk.Label(
            self.text_columns_frame,
            text="Text Column 1:",
            font=("Arial", 12),
            bg=label_bg_color,
            fg=text_color,
        )
        self.text_column1_label.grid(row=0, column=0, pady=5, padx=10)
        self.text_column1_entry = tk.Entry(
            self.text_columns_frame, width=50, font=("Arial", 12), bg=entry_bg_color
        )
        self.text_column1_entry.insert(tk.END, "Default text 1")
        self.text_column1_entry.grid(row=0, column=1, padx=5, pady=5)

        self.restore_default1_button = tk.Button(
            self.text_columns_frame,
            text="Restore to Default",
            command=lambda: self.restore_default(
                self.text_column1_entry, "Default text 1"
            ),
            bg=button_color,
            fg=button_text_color,
        )
        self.restore_default1_button.grid(row=0, column=2, padx=5, pady=5)

        # Text column 2
        self.text_column2_label = tk.Label(
            self.text_columns_frame,
            text="Text Column 2:",
            font=("Arial", 12),
            bg=label_bg_color,
            fg=text_color,
        )
        self.text_column2_label.grid(row=1, column=0, pady=5, padx=10)
        self.text_column2_entry = tk.Entry(
            self.text_columns_frame, width=50, font=("Arial", 12), bg=entry_bg_color
        )
        self.text_column2_entry.insert(tk.END, "Default text 2")
        self.text_column2_entry.grid(row=1, column=1, padx=5, pady=5)

        self.restore_default2_button = tk.Button(
            self.text_columns_frame,
            text="Restore to Default",
            command=lambda: self.restore_default(
                self.text_column2_entry, "Default text 2"
            ),
            bg=button_color,
            fg=button_text_color,
        )
        self.restore_default2_button.grid(row=1, column=2, padx=5, pady=5)

    def setup_logging(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # Create a custom logging handler to display logs in the text widget
        self.log_handler = TextWidgetHandler(self.log_text)
        self.logger.addHandler(self.log_handler)

        # Redirect stdout and stderr to our CustomText widget
        sys.stdout = self.log_handler
        sys.stderr = self.log_handler

    def show_password(self):
        if self.show_password_var.get():
            self.password_entry.config(show="")
        else:
            self.password_entry.config(show="*")

    def browse_file(self, entry):
        def browse():
            try:
                file_path = filedialog.askopenfilename(
                    initialdir="~",  # Set initial directory to user's home directory
                    filetypes=[
                        ("All files", "*.*"),  # Include all file types
                        ("CSV files", "*.csv"),
                        ("Excel files", "*.xlsx"),
                        ("Java keystore", "*.jks"),
                    ],
                )
                if file_path:
                    entry.delete(0, tk.END)
                    entry.insert(0, file_path)
                    logging.info(f"File selected: {file_path}")  # Log file selection

                    # Check file format
                    file_format = file_path.split(".")[-1].lower()
                    if file_format not in ["csv", "xlsx", "jks"]:
                        messagebox.showwarning(
                            "File Format Warning",
                            "Please select a file with one of the following formats: CSV, XLSX, JKS",
                        )
                        logging.warning(
                            f"Invalid file format selected: {file_format}"
                        )  # Log invalid file format
            except Exception as e:
                logging.error(
                    f"Error occurred while browsing file: {e}"
                )  # Log browsing error
                messagebox.showerror("Error", "An error occurred while browsing file.")

        # Start a new thread for browsing files
        threading.Thread(target=browse).start()

    def restore_default(self, entry, default_text):
        entry.delete(0, tk.END)
        entry.insert(0, default_text)

    def submit(self):
        file1_path = self.file1_entry.get()
        file2_path = self.file2_entry.get()
        password = self.password_entry.get()
        text_column1 = self.text_column1_entry.get()
        text_column2 = self.text_column2_entry.get()

        if not all([file1_path, file2_path, password]):
            logging.error("Missing input fields")  # Log missing input error
            messagebox.showerror("Error", "Please fill in all fields.")
            return None

        # Instantiate AnotherClass and pass the logger instance to it
        another_instance = AnotherClass(self.logger)
        # Pass file paths to AnotherClass for processing
        another_instance.do_something(file1_path, file2_path)

        # Simulate data processing (replace with your actual logic)
        output = f"File 1 Path: {file1_path}\nFile 2 Path: {file2_path}\nPassword: {password}\nText Column 1: {text_column1}\nText Column 2: {text_column2}"
        logging.info(output)  # Log simulated data processing

        # Show completion message
        messagebox.showinfo(
            "Process completed", "Data processing completed successfully!"
        )


root = tk.Tk()
app = App(root)
root.mainloop()
