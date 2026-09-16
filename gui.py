import os
import tkinter as tk
from tkinter import filedialog, messagebox


class ToolTip:
    """A lightweight tooltip pop-up for Tkinter widgets."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.widget.bind("<Enter>", self.show_tip)
        self.widget.bind("<Leave>", self.hide_tip)

    def show_tip(self, event=None):
        if self.tipwindow or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 25
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            background="#ffffe1",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Arial", "9", "normal"),
            padx=8,
            pady=4,
        )
        label.pack(ipadx=1)

    def hide_tip(self, event=None):
        tw = self.tipwindow
        self.tipwindow = None
        if tw:
            tw.destroy()


class PipelineGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Depth Processing Setup")
        self.root.geometry("560x320")
        self.root.resizable(True, True)  # Make setup window resizable
        self.config = None

        # Grid layout responsiveness
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)

        # Dataset Directory Field
        dir_frame = tk.Frame(root)
        dir_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 5))
        dir_frame.columnconfigure(0, weight=1)

        tk.Label(
            dir_frame, text="QUEUE Dataset Path:", font=("Arial", 10, "bold")
        ).grid(row=0, column=0, sticky="w", pady=(0, 2))

        self.path_entry = tk.Entry(dir_frame)
        self.path_entry.grid(row=1, column=0, sticky="ew", pady=5)
        self.path_entry.insert(0, "./QUEUE")

        tk.Button(dir_frame, text="Browse...", command=self.browse_folder).grid(
            row=1, column=1, padx=(5, 0), pady=5
        )

        # Parameter Input Section
        param_frame = tk.LabelFrame(
            root, text=" Parameter Settings (Hover over ❓ for details) ", padx=10, pady=10
        )
        param_frame.grid(row=2, column=0, padx=15, pady=10, sticky="nsew")
        param_frame.columnconfigure(1, weight=1)

        # 1. Alpha
        tk.Label(param_frame, text="Alpha (Learning Rate):").grid(
            row=0, column=0, sticky="w", pady=5
        )
        self.alpha_entry = tk.Entry(param_frame, width=12)
        self.alpha_entry.grid(row=0, column=1, sticky="w", pady=5, padx=(5, 5))
        self.alpha_entry.insert(0, "0.01")  #[cite: 1]
        q_alpha = tk.Label(
            param_frame, text="❓", cursor="question_arrow", fg="#0066cc", font=("Arial", 10, "bold")
        )
        q_alpha.grid(row=0, column=2, sticky="w", pady=5)
        ToolTip(
            q_alpha,
            "Alpha controls background update speed.\n"
            "• Lower values (e.g. 0.01) keep slow or standing people from\n"
            "  blending into the background model over time.",
        )

        # 2. Delta
        tk.Label(param_frame, text="Delta Threshold (mm):").grid(
            row=1, column=0, sticky="w", pady=5
        )
        self.delta_entry = tk.Entry(param_frame, width=12)
        self.delta_entry.grid(row=1, column=1, sticky="w", pady=5, padx=(5, 5))
        self.delta_entry.insert(0, "200.0")  #[cite: 1]
        q_delta = tk.Label(
            param_frame, text="❓", cursor="question_arrow", fg="#0066cc", font=("Arial", 10, "bold")
        )
        q_delta.grid(row=1, column=2, sticky="w", pady=5)
        ToolTip(
            q_delta,
            "Delta is the depth difference threshold to detect moving targets.\n"
            "• 200.0 mm (20 cm) means an object must be at least 20 cm closer\n"
            "  than the background plane to be detected.",
        )

        # 3. Sobel Edge Threshold
        tk.Label(param_frame, text="Sobel Edge Threshold (mm):").grid(
            row=2, column=0, sticky="w", pady=5
        )
        self.sobel_entry = tk.Entry(param_frame, width=12)
        self.sobel_entry.grid(row=2, column=1, sticky="w", pady=5, padx=(5, 5))
        self.sobel_entry.insert(0, "800.0")
        q_sobel = tk.Label(
            param_frame, text="❓", cursor="question_arrow", fg="#0066cc", font=("Arial", 10, "bold")
        )
        q_sobel.grid(row=2, column=2, sticky="w", pady=5)
        ToolTip(
            q_sobel,
            "Sobel edge threshold detects depth discontinuities.\n"
            "• Used to draw cutting boundaries and separate people\n"
            "  walking closely together in a crowd.",
        )

        # Start Button
        tk.Button(
            root,
            text="Start Processing",
            bg="#2b8cbe",
            fg="white",
            font=("Arial", 10, "bold"),
            command=self.submit,
        ).grid(row=3, column=0, pady=(0, 15))

    def browse_folder(self):
        folder_selected = filedialog.askdirectory(title="Select QUEUE Directory")
        if folder_selected:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, folder_selected)

    def submit(self):
        data_dir = self.path_entry.get().strip()

        try:
            alpha = float(self.alpha_entry.get())
            delta = float(self.delta_entry.get())
            sobel_thresh = float(self.sobel_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "All parameters must be numbers.")
            return

        depth_dir = os.path.join(data_dir, "depth")
        rgb_dir = os.path.join(data_dir, "RGB")

        if not os.path.exists(depth_dir) or not os.path.exists(rgb_dir):
            messagebox.showerror(
                "Path Error", f"Missing 'depth' or 'RGB' subfolders in:\n{data_dir}"
            )
            return

        self.config = {
            "data_dir": data_dir,
            "alpha": alpha,
            "delta": delta,
            "sobel_thresh": sobel_thresh,
        }
        self.root.destroy()


def launch_gui():
    root = tk.Tk()
    app = PipelineGUI(root)
    root.mainloop()
    return app.config