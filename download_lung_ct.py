import os
import pydicom
import shutil
from pydicom.data import get_testdata_files

def generate_negatives(out_dir="Dataset/LungCT_Negatives"):
    os.makedirs(out_dir, exist_ok=True)
    print("Extracting sample DICOMs from pydicom's test data cache...")
    
    # Get all sample DICOM files built into pydicom
    dcm_files = [f for f in get_testdata_files("*") if f.endswith(".dcm") or f.endswith(".dcm")]
    
    count = 0
    for f in dcm_files:
        try:
            # Only copy files that have pixel data
            ds = pydicom.dcmread(f)
            if hasattr(ds, "PixelData"):
                out_path = os.path.join(out_dir, f"neg_{count}_{os.path.basename(f)}")
                shutil.copy(f, out_path)
                count += 1
        except Exception:
            pass
            
    print(f"Successfully copied {count} medical images (MRIs, random CTs, etc.) into {out_dir}")
    print("These will serve as robust Negative Pairs for STEGO.")

if __name__ == "__main__":
    generate_negatives()
