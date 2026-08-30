#!/usr/bin/env python3
"""
AI Retouch Pro - Standalone Python Desktop Studio Application
High-End Dark UI with Before/After Canvas, Precision Sliders, Preset Profiles, and Batch Processing.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np

# Add backend directory to sys.path
sys.path.insert(0, os.path.dirname(__file__))

from face_segmenter import segment_face_skin
from pimple_detector_v2 import detect_pimple_candidates
from skin_smoother import apply_frequency_separation
from skin_toner import apply_relative_lighten
from dodge_and_burn import apply_ai_dodge_and_burn
from eye_teeth_enhancer import apply_eye_and_teeth_enhancement
from shine_neutralizer import apply_shine_neutralizer
from spot_classifier import filter_blobs_by_preference, classify_spot

try:
    from simple_lama_inpainting import SimpleLama
    LAMA_AVAILABLE = True
except Exception:
    LAMA_AVAILABLE = False


class AIRetouchStudioApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Retouch Pro - Python Studio Suite")
        self.geometry("1180x780")
        self.minsize(980, 680)
        self.configure(bg="#14161A")

        self.original_pil = None
        self.processed_pil = None
        self.display_pil = None
        self.lama_model = None
        self.current_scale = 1.0

        self.setup_styles()
        self.build_ui()
        self.load_models_async()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        # Color Palette
        bg_dark = "#14161A"
        card_bg = "#1D2128"
        accent_blue = "#2A72E5"
        text_white = "#F0F4F8"
        text_muted = "#9DA7B3"

        style.configure(".", background=bg_dark, foreground=text_white, font=("Segoe UI", 9))
        style.configure("TFrame", background=bg_dark)
        style.configure("Card.TFrame", background=card_bg, relief="flat")
        style.configure("TLabel", background=bg_dark, foreground=text_white)
        style.configure("Card.TLabel", background=card_bg, foreground=text_white)
        style.configure("Muted.TLabel", background=card_bg, foreground=text_muted, font=("Segoe UI", 8))
        style.configure("Header.TLabel", background=bg_dark, foreground="#FFFFFF", font=("Segoe UI", 12, "bold"))
        
        style.configure("Primary.TButton", font=("Segoe UI", 10, "bold"), background=accent_blue, foreground="#FFFFFF", padding=6)
        style.map("Primary.TButton", background=[("active", "#1E58B8"), ("pressed", "#164491")])

        style.configure("Tool.TButton", font=("Segoe UI", 9), background="#282E38", foreground="#FFFFFF", padding=4)
        style.map("Tool.TButton", background=[("active", "#363E4B"), ("pressed", "#20252D")])

        style.configure("TNotebook", background=bg_dark, borderwidth=0)
        style.configure("TNotebook.Tab", background="#21262E", foreground=text_muted, padding=[12, 5], font=("Segoe UI", 9, "bold"))
        style.map("TNotebook.Tab", background=[("selected", card_bg)], foreground=[("selected", text_white)])

    def build_ui(self):
        # Top App Bar
        top_bar = ttk.Frame(self, padding=(16, 12, 16, 8))
        top_bar.pack(fill="x")
        
        ttk.Label(top_bar, text="AI RETOUCH PRO  |  STUDIO SUITE", style="Header.TLabel").pack(side="left")
        
        self.lbl_status_badge = ttk.Label(top_bar, text="[INITIALIZING ENGINE...]", font=("Segoe UI", 9, "bold"), foreground="#F59E0B")
        self.lbl_status_badge.pack(side="right")

        # Main Layout: Left Canvas (Split/Viewer) + Right Sidebar Controls
        main_content = ttk.Frame(self, padding=(12, 0, 12, 8))
        main_content.pack(fill="both", expand=True)

        # Left Canvas Panel
        canvas_card = ttk.Frame(main_content, style="Card.TFrame", padding=8)
        canvas_card.pack(side="left", fill="both", expand=True, padx=(0, 8))

        # Canvas Toolbar (Open, Save, Split Slider, Reset)
        canvas_tools = ttk.Frame(canvas_card, style="Card.TFrame")
        canvas_tools.pack(fill="x", pady=(0, 6))

        ttk.Button(canvas_tools, text="Open Image", style="Tool.TButton", command=self.on_open_image).pack(side="left", padx=3)
        ttk.Button(canvas_tools, text="Save Result", style="Tool.TButton", command=self.on_save_image).pack(side="left", padx=3)
        ttk.Button(canvas_tools, text="Batch Process", style="Tool.TButton", command=self.on_batch_process).pack(side="left", padx=3)
        ttk.Button(canvas_tools, text="Original View", style="Tool.TButton", command=self.show_original).pack(side="left", padx=3)
        ttk.Button(canvas_tools, text="Retouched View", style="Tool.TButton", command=self.show_processed).pack(side="left", padx=3)

        self.lbl_zoom = ttk.Label(canvas_tools, text="100%", style="Muted.TLabel")
        self.lbl_zoom.pack(side="right", padx=6)

        # Image Canvas
        self.canvas = tk.Canvas(canvas_card, bg="#0E1013", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw_canvas())

        # Right Sidebar Controls
        sidebar = ttk.Frame(main_content, style="Card.TFrame", width=380, padding=12)
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)

        # Presets Group
        ttk.Label(sidebar, text="Studio Preset Profile", style="Card.TLabel", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        
        self.preset_var = tk.StringVar(value="Natural Studio (Subtle & Balanced)")
        self.ddl_preset = ttk.Combobox(
            sidebar, 
            textvariable=self.preset_var,
            values=[
                "Natural Studio (Subtle & Balanced)",
                "Glamour Velvet (Smooth Beauty)",
                "Deep Blemish & Acne Fix",
                "High-End Editorial (Pore Preserved)",
                "Custom Configuration"
            ],
            state="readonly"
        )
        self.ddl_preset.pack(fill="x", pady=(0, 6))
        self.ddl_preset.bind("<<ComboboxSelected>>", self.on_preset_change)

        self.lbl_preset_desc = ttk.Label(
            sidebar, 
            text="Natural Studio: Balanced skin evening with 100% natural pore relief & gentle Dodge/Burn.",
            style="Muted.TLabel",
            wraplength=350
        )
        self.lbl_preset_desc.pack(fill="x", pady=(0, 10))

        # Primary Master Button
        self.btn_master = ttk.Button(sidebar, text="EXECUTE COMPLETE STUDIO RETOUCH", style="Primary.TButton", command=self.run_master_retouch)
        self.btn_master.pack(fill="x", pady=(0, 12))

        # Tabs for Tools vs Fine Sliders
        tabs = ttk.Notebook(sidebar)
        tabs.pack(fill="both", expand=True)

        # Tab A: Individual Studio Tools
        tab_tools = ttk.Frame(tabs, style="Card.TFrame", padding=8)
        tabs.add(tab_tools, text="Studio Modules")

        ttk.Button(tab_tools, text="Heal Blemishes & Acne", style="Tool.TButton", command=lambda: self.run_single_module("heal")).pack(fill="x", pady=3)
        ttk.Button(tab_tools, text="Smooth Skin (Tri-Band)", style="Tool.TButton", command=lambda: self.run_single_module("smooth")).pack(fill="x", pady=3)
        ttk.Button(tab_tools, text="Tone Lightening", style="Tool.TButton", command=lambda: self.run_single_module("lighten")).pack(fill="x", pady=3)
        ttk.Button(tab_tools, text="Dodge & Burn Contours", style="Tool.TButton", command=lambda: self.run_single_module("db")).pack(fill="x", pady=3)
        ttk.Button(tab_tools, text="Eyes & Teeth Enhancement", style="Tool.TButton", command=lambda: self.run_single_module("eye_teeth")).pack(fill="x", pady=3)
        ttk.Button(tab_tools, text="Shine Neutralizer", style="Tool.TButton", command=lambda: self.run_single_module("shine")).pack(fill="x", pady=3)
        ttk.Button(tab_tools, text="Preview Blemish Detection Mask", style="Tool.TButton", command=lambda: self.run_single_module("mask")).pack(fill="x", pady=(8, 3))

        # Tab B: Precision Sliders
        tab_sliders = ttk.Frame(tabs, style="Card.TFrame", padding=8)
        tabs.add(tab_sliders, text="Precision Sliders")

        self.sld_sens = self.create_slider_row(tab_sliders, "Blemish Sensitivity:", 40, 10, 100)
        self.sld_smooth = self.create_slider_row(tab_sliders, "Skin Smoothing:", 35, 5, 100)
        self.sld_tex = self.create_slider_row(tab_sliders, "Texture / Pores:", 45, 0, 100)
        self.sld_str = self.create_slider_row(tab_sliders, "Tone Lighten:", 25, 0, 100)
        self.sld_db = self.create_slider_row(tab_sliders, "Dodge & Burn:", 35, 0, 100)
        self.sld_shine = self.create_slider_row(tab_sliders, "Shine Reducer:", 30, 0, 100)
        self.sld_et = self.create_slider_row(tab_sliders, "Eyes & Teeth Lift:", 30, 0, 100)

        # Bottom Progress Bar & Info
        bottom_bar = ttk.Frame(self, padding=(16, 6, 16, 10))
        bottom_bar.pack(fill="x")
        
        self.progress_var = tk.DoubleVar(value=0)
        self.prg_bar = ttk.Progressbar(bottom_bar, variable=self.progress_var, maximum=100)
        self.prg_bar.pack(fill="x", side="left", expand=True, padx=(0, 12))

        self.lbl_progress_status = ttk.Label(bottom_bar, text="Ready.", style="Muted.TLabel")
        self.lbl_progress_status.pack(side="right")

    def create_slider_row(self, parent, label_text, default_val, min_val, max_val):
        frame = ttk.Frame(parent, style="Card.TFrame")
        frame.pack(fill="x", pady=3)

        hdr = ttk.Frame(frame, style="Card.TFrame")
        hdr.pack(fill="x")
        ttk.Label(hdr, text=label_text, style="Card.TLabel").pack(side="left")
        lbl_val = ttk.Label(hdr, text=f"{int(default_val)}%", style="Card.TLabel")
        lbl_val.pack(side="right")

        var = tk.DoubleVar(value=default_val)
        
        def on_change(v):
            lbl_val.config(text=f"{int(float(v))}%")
            self.preset_var.set("Custom Configuration")
            self.lbl_preset_desc.config(text="Custom user configuration.")

        sld = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, command=on_change)
        sld.pack(fill="x", pady=(1, 4))
        return var

    def load_models_async(self):
        def _loader():
            try:
                if LAMA_AVAILABLE:
                    self.lama_model = SimpleLama()
                self.lbl_status_badge.config(text="[ONLINE: PYTORCH LOCAL ENGINE]", foreground="#10B981")
            except Exception as e:
                self.lbl_status_badge.config(text=f"[ENGINE LOADED: OPENCV ONLY]", foreground="#F59E0B")

        threading.Thread(target=_loader, daemon=True).start()

    def on_preset_change(self, event=None):
        idx = self.ddl_preset.current()
        presets = [
            (40, 35, 45, 25, 35, 30, 30, "Natural Studio: Balanced skin evening with 100% natural pore relief & gentle Dodge/Burn."),
            (60, 65, 25, 45, 50, 50, 50, "Glamour Velvet: Soft beauty smoothing with radiant luminance & high vitality."),
            (80, 45, 35, 20, 40, 30, 25, "Deep Acne Fix: Aggressive neural inpainting for deep acne & redness calming."),
            (45, 30, 60, 30, 60, 35, 40, "High-End Editorial: Sculpted facial micro-contours with 100% micro-pore retention."),
            (40, 35, 45, 25, 35, 30, 30, "Custom Configuration: Fine manual parameter adjustments.")
        ]
        if 0 <= idx < len(presets):
            s, sm, t, l, db, sh, et, desc = presets[idx]
            self.sld_sens.set(s)
            self.sld_smooth.set(sm)
            self.sld_tex.set(t)
            self.sld_str.set(l)
            self.sld_db.set(db)
            self.sld_shine.set(sh)
            self.sld_et.set(et)
            self.lbl_preset_desc.config(text=desc)

    def on_open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp *.bmp *.tiff")])
        if path and os.path.exists(path):
            self.original_pil = Image.open(path).convert("RGB")
            self.processed_pil = self.original_pil.copy()
            self.display_pil = self.processed_pil
            self.redraw_canvas()
            self.lbl_progress_status.config(text=f"Loaded: {os.path.basename(path)} ({self.original_pil.width}x{self.original_pil.height})")

    def on_save_image(self):
        if not self.processed_pil:
            messagebox.showinfo("No Image", "Please open and retouch an image first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg")])
        if path:
            self.processed_pil.save(path)
            messagebox.showinfo("Saved", f"Retouched portrait saved successfully to:\n{path}")

    def on_batch_process(self):
        paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.webp")])
        if not paths:
            return
        out_dir = filedialog.askdirectory(title="Select Output Folder")
        if not out_dir:
            return

        def _batch_worker():
            total = len(paths)
            for i, p in enumerate(paths):
                self.lbl_progress_status.config(text=f"Batch {i+1}/{total}: {os.path.basename(p)}")
                self.progress_var.set(int(((i + 1) / total) * 100))
                try:
                    img = Image.open(p).convert("RGB")
                    retouched = self.process_image_full(img)
                    out_path = os.path.join(out_dir, "retouched_" + os.path.basename(p))
                    retouched.save(out_path)
                except Exception as e:
                    print(f"Failed {p}: {e}")
            self.lbl_progress_status.config(text=f"Batch Complete: {total} images saved.")
            messagebox.showinfo("Batch Complete", f"Successfully retouched {total} portraits to:\n{out_dir}")

        threading.Thread(target=_batch_worker, daemon=True).start()

    def show_original(self):
        if self.original_pil:
            self.display_pil = self.original_pil
            self.redraw_canvas()

    def show_processed(self):
        if self.processed_pil:
            self.display_pil = self.processed_pil
            self.redraw_canvas()

    def redraw_canvas(self):
        if not self.display_pil:
            return
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            return

        img = self.display_pil.copy()
        img.thumbnail((cw, ch), Image.Resampling.LANCZOS)
        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        x = (cw - img.width) // 2
        y = (ch - img.height) // 2
        self.canvas.create_image(x, y, anchor="nw", image=self.tk_image)

    def process_image_full(self, img_pil: Image.Image) -> Image.Image:
        import cv2
        img_np = np.array(img_pil)
        skin_mask, _ = segment_face_skin(img_np)

        sens = self.sld_sens.get() / 100.0
        smooth = self.sld_smooth.get() / 100.0
        tex = self.sld_tex.get() / 100.0
        light = self.sld_str.get() / 100.0
        db = self.sld_db.get() / 100.0
        shine = self.sld_shine.get() / 100.0
        et = self.sld_et.get() / 100.0

        cur_img = img_np.copy()

        # 1. Auto-Heal Blemishes (Fitzpatrick & Melanin Preserved)
        blobs, p_mask = detect_pimple_candidates(cur_img, skin_mask, sensitivity=sens)
        if len(blobs) > 0:
            active_blobs = filter_blobs_by_preference(blobs, cur_img, preserve_moles=True, preserve_freckles=False)
            if active_blobs:
                h_mask = np.zeros(cur_img.shape[:2], dtype=np.uint8)
                for b in active_blobs:
                    x = int(b.get("x", b.get("cx", 0)))
                    y = int(b.get("y", b.get("cy", 0)))
                    r = int(b.get("radius", b.get("r", 6)))
                    cv2.circle(h_mask, (x, y), max(4, int(r * 1.3)), 255, -1)
                if self.lama_model:
                    cur_img = np.array(self.lama_model(Image.fromarray(cur_img), Image.fromarray(h_mask)))
                else:
                    cur_img = cv2.inpaint(cur_img, h_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        # 2. Dodge & Burn
        if db > 0.05:
            cur_img = apply_ai_dodge_and_burn(cur_img, skin_mask, strength=db)

        # 3. Smooth Skin
        if smooth > 0.05:
            cur_img = apply_frequency_separation(cur_img, skin_mask, strength=smooth, texture_keep=tex)

        # 4. Lighten
        if light > 0.05:
            cur_img = apply_relative_lighten(cur_img, skin_mask, strength=light)

        # 5. Eyes & Teeth
        if et > 0.05:
            cur_img = apply_eye_and_teeth_enhancement(cur_img, teeth_whiten=et, eye_brighten=et)

        # 6. Shine
        if shine > 0.05:
            cur_img = apply_shine_neutralizer(cur_img, skin_mask, strength=shine)

        return Image.fromarray(cur_img)

    def run_master_retouch(self):
        if not self.original_pil:
            messagebox.showinfo("Open Image", "Please open a portrait photograph first.")
            return

        def _worker():
            self.btn_master.config(state="disabled")
            self.lbl_progress_status.config(text="Processing master studio retouch pipeline...")
            self.progress_var.set(30)
            
            try:
                res = self.process_image_full(self.original_pil)
                self.processed_pil = res
                self.display_pil = self.processed_pil
                self.progress_var.set(100)
                self.lbl_progress_status.config(text="Master Retouch Completed Successfully.")
                self.after(0, self.redraw_canvas)
            except Exception as e:
                messagebox.showerror("Retouch Error", f"Error during processing: {e}")
                self.lbl_progress_status.config(text="Processing Failed.")
            finally:
                self.btn_master.config(state="normal")

        threading.Thread(target=_worker, daemon=True).start()

    def run_single_module(self, mod_type: str):
        if not self.original_pil:
            messagebox.showinfo("Open Image", "Please open a portrait photograph first.")
            return

        def _worker():
            import cv2
            self.lbl_progress_status.config(text=f"Executing {mod_type}...")
            self.progress_var.set(40)
            img_np = np.array(self.original_pil)
            skin_mask, _ = segment_face_skin(img_np)
            
            if mod_type == "heal":
                sens = self.sld_sens.get() / 100.0
                blobs, _ = detect_pimple_candidates(img_np, skin_mask, sensitivity=sens)
                if blobs:
                    active_blobs = filter_blobs_by_preference(blobs, img_np, preserve_moles=True, preserve_freckles=False)
                    if active_blobs:
                        h_mask = np.zeros(img_np.shape[:2], dtype=np.uint8)
                        for b in active_blobs:
                            x = int(b.get("x", b.get("cx", 0)))
                            y = int(b.get("y", b.get("cy", 0)))
                            r = int(b.get("radius", b.get("r", 6)))
                            cv2.circle(h_mask, (x, y), max(4, int(r * 1.3)), 255, -1)
                        if self.lama_model:
                            res_np = np.array(self.lama_model(self.original_pil, Image.fromarray(h_mask)))
                        else:
                            res_np = cv2.inpaint(img_np, h_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
                    else:
                        res_np = img_np
                else:
                    res_np = img_np
            elif mod_type == "smooth":
                res_np = apply_frequency_separation(img_np, skin_mask, strength=self.sld_smooth.get()/100.0, texture_keep=self.sld_tex.get()/100.0)
            elif mod_type == "lighten":
                res_np = apply_relative_lighten(img_np, skin_mask, strength=self.sld_str.get()/100.0)
            elif mod_type == "db":
                res_np = apply_ai_dodge_and_burn(img_np, skin_mask, strength=self.sld_db.get()/100.0)
            elif mod_type == "eye_teeth":
                et = self.sld_et.get()/100.0
                res_np = apply_eye_and_teeth_enhancement(img_np, teeth_whiten=et, eye_brighten=et)
            elif mod_type == "shine":
                res_np = apply_shine_neutralizer(img_np, skin_mask, strength=self.sld_shine.get()/100.0)
            elif mod_type == "mask":
                _, p_mask = detect_pimple_candidates(img_np, skin_mask, sensitivity=self.sld_sens.get()/100.0)
                r = Image.new("L", self.original_pil.size, 255)
                g = Image.new("L", self.original_pil.size, 40)
                b = Image.new("L", self.original_pil.size, 40)
                rgba_mask = Image.merge("RGBA", (r, g, b, Image.fromarray(p_mask)))
                comp = self.original_pil.copy().convert("RGBA")
                comp.alpha_composite(rgba_mask)
                res_np = np.array(comp.convert("RGB"))

            self.processed_pil = Image.fromarray(res_np)
            self.display_pil = self.processed_pil
            self.progress_var.set(100)
            self.lbl_progress_status.config(text=f"{mod_type.upper()} Finished.")
            self.after(0, self.redraw_canvas)

        threading.Thread(target=_worker, daemon=True).start()


if __name__ == "__main__":
    app = AIRetouchStudioApp()
    app.mainloop()
