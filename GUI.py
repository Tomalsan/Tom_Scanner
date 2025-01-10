import tkinter as tk
from tkinter import filedialog, messagebox,ttk
from PIL import Image, ImageTk
from large_cont import *

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((400, 400))
        img = ImageTk.PhotoImage(image)
       
        show_image_page(file_path)
        current_images.append(file_path)
        image_name = os.path.basename(file_path)
        
        print(f"image path",file_path)

def show_image_page(img_path):
    global img_tk 

    
    for widget in root.winfo_children():
        widget.destroy()

    
    image = Image.open(img_path)
    img_width, img_height = image.size
    max_width, max_height = 800, 800
    scale = min(max_width / img_width, max_height / img_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    image = image.resize((new_width, new_height), Image.LANCZOS)

    img_tk = ImageTk.PhotoImage(image)

    
    canvas_frame = tk.Frame(root)
    canvas_frame.pack(fill=tk.BOTH, expand=True)

   
    canvas = tk.Canvas(canvas_frame)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    scrollbar_y = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
    scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

   
    scrollbar_x = tk.Scrollbar(root, orient=tk.HORIZONTAL, command=canvas.xview)
    scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)

    canvas.config(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)

   
    canvas.create_image(0, 0, anchor='nw', image=img_tk)
    canvas.image = img_tk

    
    canvas.config(scrollregion=canvas.bbox('all'))

    
    back_button = tk.Button(root, text="Back", command=main_page, font=("Arial", 14), width=10, bg="#f0ad4e", fg="white")
    back_button.pack(pady=10)

    
    scan_button = tk.Button(root, text="Scan", command=lambda: scan_image(img_path), font=("Arial", 14), width=10, bg="#5bc0de", fg="white")
    scan_button.pack(pady=10)


def scan_image(img_path):
    draw_bounding_box(img_path)
    
        
def make_pdf():
    folder_path = filedialog.askdirectory(title="Select Output Folder")
    if folder_path:
        images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        
        if images:
            selected_images = filedialog.askopenfilenames(initialdir=folder_path, title="Select Images in Order",
                                                          filetypes=[("Image files", "*.png *.jpg *.jpeg")])
            
            if selected_images:
                pdf_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
                if pdf_path:
                    try:
                        image_objects = [Image.open(img).convert('RGB') for img in selected_images]
                        if image_objects:
                            
                            sorted_images = sorted(zip(selected_images, image_objects), key=lambda x: selected_images.index(x[0]))
                            sorted_images = [img_obj for _, img_obj in sorted_images]
                            
                            sorted_images[0].save(pdf_path, save_all=True, append_images=sorted_images[1:])
                            messagebox.showinfo("Success", "PDF created successfully!")
                        else:
                            messagebox.showwarning("No Images", "No valid images to create PDF.")
                    except PermissionError:
                        messagebox.showerror("Error", "Permission denied. Please close any open PDF files and try again.")
            else:
                messagebox.showwarning("No Selection", "No images selected.")
        else:
            messagebox.showwarning("No Images", "No images found in the folder.")
    else:
        messagebox.showwarning("No Folder", "No folder selected.")


def delete_images():
    folder_path = filedialog.askdirectory(title="Select Folder to Delete Images From")
    if folder_path:
        selected_images = filedialog.askopenfilenames(initialdir=folder_path, title="Select Images to Delete",
                                                      filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if selected_images:
            for image_path in selected_images:
                try:
                    os.remove(image_path)
                    messagebox.showinfo("Deleted", f"Deleted: {image_path}")
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to delete: {image_path}\nError: {str(e)}")
            messagebox.showinfo("Batch Delete", "All selected images deleted.")
        else:
            messagebox.showwarning("No Selection", "No images selected.")
    else:
        messagebox.showwarning("No Folder", "No folder selected.")

def main_page():
    for widget in root.winfo_children():
        widget.destroy()
        
    title_label = tk.Label(root, text="TOM_SCANNER", font=("Arial", 24))
    title_label.pack(pady=30)
    
    frame = tk.Frame(root)
    frame.pack(pady=30)

    open_button = tk.Button(frame, text="Open Image", command=open_image, font=("Arial", 14), width=20, height=2, bg="#4CAF50", fg="white")
    open_button.pack(pady=15)

    scan_button = tk.Button(frame, text="Make PDF", command=make_pdf, font=("Arial", 14), width=20, height=2, bg="#2196F3", fg="white")
    scan_button.pack(pady=15)

    delete_button = tk.Button(frame, text="Delete Images", command=delete_images, font=("Arial", 14), width=20, height=2, bg="#f44336", fg="white")
    delete_button.pack(pady=15)

root = tk.Tk()
root.title("TomSCANNER")
current_images = []

main_page()

root.mainloop()
