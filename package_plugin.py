import os
import zipfile
import shutil

def package_ccx():
    plugin_dir = os.path.join(os.path.dirname(__file__), "uxp-plugin")
    output_ccx = os.path.join(os.path.dirname(__file__), "AI-Blemish-Remover.ccx")
    
    print(f"[*] Packaging UXP plugin from: {plugin_dir}")
    print(f"[*] Output .ccx file: {output_ccx}")
    
    # Create zip / ccx archive
    with zipfile.ZipFile(output_ccx, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(plugin_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Compute relative path inside the plugin folder
                rel_path = os.path.relpath(file_path, plugin_dir)
                zipf.write(file_path, rel_path)
                print(f"  + Added {rel_path}")

    print(f"\n[+] Successfully created: {output_ccx} ({os.path.getsize(output_ccx)} bytes)")
    print("[*] You can double-click this .ccx file to install it into Photoshop!")

if __name__ == "__main__":
    package_ccx()
